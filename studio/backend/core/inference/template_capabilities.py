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
            self.terminated,
            self.continued,
            self.budget,
        )


def _field(node):
    if isinstance(node, nodes.Getattr):
        return node.attr
    if isinstance(node, nodes.Getitem) and isinstance(node.arg, nodes.Const):
        return node.arg.value
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


def _as_const(node, state):
    """The literal a node is known to be, either written out or bound to a name."""
    if isinstance(node, nodes.Const):
        return node
    if (
        state is not None
        and isinstance(node, nodes.Name)
        and node.name in state.consts
    ):
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
                [
                    nodes.Operand(operand.op, value)
                    for operand, value in zip(node.ops, operands)
                ],
            )
            try:
                # The rebuilt node is synthetic, so it carries no environment of its
                # own and has to be handed an evaluation context explicitly.
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
                expression: fact
                for expression, fact in state.facts.items()
                if fact[1] & live
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


def _template_built(node, state):
    """The member-name shortcuts below read a field off an untracked value, so they
    only mean anything when the base is one. A base the template constructed is
    tracked, and its provenance already lives in `aliases`."""
    return _reference_key(node.node) in state.constructed


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
    if _non_empty_tools(node, state):
        return True
    if isinstance(node, nodes.Compare) and len(node.ops) == 1:
        operand = node.ops[0]
        if operand.op == "eq":
            return any(
                _field(role) == "role"
                and not _template_built(role, state)
                and isinstance(value, nodes.Const)
                and value.value == "tool"
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
    if isinstance(value, nodes.Call) and isinstance(value.node, nodes.Getattr):
        # `catalog.pop('schema')` evaluates to whatever sat at that field.
        removed = _removed_key(value.node.attr, value)
        if removed is not None:
            return _select(_value_aliases(value.node.node, state, active), removed)
        # A shallow copy is the receiver again as far as provenance goes; without this
        # the fallback reads `copy` as a data field and loses everything under it.
        if value.node.attr == "copy" and not value.args and not value.kwargs:
            return _value_aliases(value.node.node, state, active)
    if isinstance(value, nodes.Call) and isinstance(value.node, nodes.Name):
        if value.node.name in ("namespace", "dict"):
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
            # The macro's own names stay live: nothing outside it constrains its body.
            emits, children = _scan(
                macro.body, local, active | {macro.name}, tail = _names(macro)
            )
            # A namespace write inside a macro escapes it, so the caller sees it:
            # {% macro load() %}{% set ns.catalog = tools %}{% endmacro %}{{ load() }}
            # leaves the catalog in ns. _export_scope already knows which of a
            # scope's mutations outlive it, and the macro's own parameters were
            # dropped from `assigned` by the scoped copy above.
            for child in children:
                exported = _export_scope(state, child)
                state.aliases.clear()
                state.aliases.update(exported.aliases)
                state.mutated.update(exported.mutated)
                state.facts = exported.facts
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
        if _constructs_object(value):
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
            state.constructed.discard(key)
    if isinstance(target, nodes.Name):
        if isinstance(value, nodes.Const) and isinstance(value.value, (str, int)):
            state.consts[target.name] = value.value
        else:
            state.consts.pop(target.name, None)
    truth = _constant_truth(value, source) if value is not None else None
    if isinstance(target, nodes.Name) and truth is not None:
        state.facts[repr(nodes.Name(target.name, "load"))] = (truth, {target.name})


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


def _constructs_object(value):
    if isinstance(value, (nodes.Dict, nodes.List, nodes.Tuple)):
        return True
    return (
        isinstance(value, nodes.Call)
        and isinstance(value.node, nodes.Name)
        and value.node.name == "namespace"
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
    positional = {suffix for arg in call.args for suffix in _value_aliases(arg, state, active)}
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
    removed = _removed_key(method, call)
    if method != "clear" and removed is None and not paths:
        return
    if method == "clear" or removed is _UNKNOWN:
        _replace(state.aliases, key, set())
    elif removed is not None:
        # pop/remove/discard take the value back out, so its provenance goes too.
        _replace(state.aliases, (*key, removed), set())
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


def _invokes_caller(call, state):
    """Whether the macro a `{% call %}` targets actually runs its caller block."""
    macro = state.macros.get(call.node.name) if isinstance(call.node, nodes.Name) else None
    if macro is None:
        # An unknown callee could invoke it, so assume the block runs.
        return True
    return any(
        isinstance(found.node, nodes.Name) and found.node.name == "caller"
        for statement in macro.body
        for found in statement.find_all(nodes.Call)
    )


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
                _bind_paths(
                    node.target, _select(_value_aliases(node.iter, parent, active), _UNKNOWN), local
                )
            else:
                _bind(node.target, value, local, active)
            # `loop.first` reprs the same in every loop, so an outer loop's facts would
            # otherwise prune branches of a nested one.
            _forget(("loop",), local)
            candidates = [local] if node.test is None else _assume(node.test, True, local)
            for candidate in candidates:
                emits, children = _scan(
                    node.body,
                    candidate,
                    active,
                    guarded or _tool_reference(node.iter, parent),
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
    if not literal or (node.test is not None and else_reachable):
        emits, children = _scan(node.else_, state.copy(scoped = True), active, guarded, inner_tail)
        if emits:
            return True, []
        states.extend(_export_scope(state, child) for child in children)
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
                if any(
                    _is_payload(value) and (guarded or _value_aliases(value, current, active))
                    for value in node.nodes
                ):
                    return True, []
            elif isinstance(node, nodes.Assign):
                # `{% set _ = xs.append(...) %}` is how templates mutate without the do
                # extension, so the call mutates even though this is an assignment.
                # Bind first: `{% set x = catalog.pop('schema') %}` hands x the value the
                # mutation is about to take out of catalog.
                _bind(node.target, node.node, current, active)
                _mutate(node.node, current, active)
            elif isinstance(node, nodes.ExprStmt):
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
                if _value_aliases(node.call, current, active):
                    return True, []
                # The caller block runs only if the macro invokes caller().
                if _invokes_caller(node.call, current):
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
                emits, _ = _scan(node.body, current.copy(scoped = True), active, guarded, rest)
                _bind_paths(node.target, {()} if emits else set(), current)
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
        emits, _ = _scan(tree.body, _State({("tools",), ("tool_calls",)}), set())
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
