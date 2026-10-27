# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GRPO trainer side of unsloth_zoo's MoE routing replay (UNSLOTH_GRPO_ROUTING_REPLAY=1).

Only used when routing replay is requested. Two things the zoo hooks need from the trainer:
- their routing weight formula is picked from the first replayed router call, which reads tensor
  values; done inside a compiled region that is a graph break (fullgraph fails), so it is picked
  here once, eagerly, right after the hooks are installed;
- PEFT reference scoring (`disable_adapter()`, or a named reference adapter made active with
  `set_adapter()`) must see the reference's own routing, as a separate reference model does, so the
  hooks are detached while it scores.
"""

import contextlib
import inspect
import warnings

import torch

__all__ = ["routing_replay_hook_snapshot", "prepare_routing_replay"]

_HOOK_DICTS = (
    ("_forward_hooks", ("_forward_hooks_with_kwargs", "_forward_hooks_always_called")),
    ("_forward_pre_hooks", ("_forward_pre_hooks_with_kwargs",)),
)


def routing_replay_hook_snapshot(model):
    """Hook ids on every module of `model`, taken before install_routing_replay."""
    return {
        id(module): tuple(set(getattr(module, name, {})) for name, _ in _HOOK_DICTS)
        for module in model.modules()
    }


def _new_hooks(model, before):
    hooks = []
    for module in model.modules():
        old = before.get(id(module), (set(), set()))
        for (name, extras), seen in zip(_HOOK_DICTS, old):
            for hook_id in getattr(module, name, {}):
                if hook_id not in seen:
                    hooks.append((module, name, extras, hook_id))
    return hooks


def _detach(hooks):
    # Keep each table's key order: ids do not give call order (prepend = True).
    orders = {}
    removed = []
    for module, name, extras, hook_id in hooks:
        for table in [getattr(module, name, None)] + [getattr(module, e, None) for e in extras]:
            if table is None or hook_id not in table:
                continue
            orders.setdefault(id(table), (table, list(table.keys())))
            removed.append((table, hook_id, table.pop(hook_id)))
    return orders, removed


def _reattach(detached):
    orders, removed = detached
    for table, hook_id, value in removed:
        table[hook_id] = value
    for table, order in orders.values():
        known = set(order)
        added = [k for k in table if k not in known]  # registered while detached: keep last
        items = dict(table)
        table.clear()
        for k in order + added:
            if k in items:
                table[k] = items[k]


class _ReplayDetacher:
    """Replay hooks stay off while any reason holds (disabled adapter or non-policy set_adapter)."""

    def __init__(self, hooks):
        self.hooks = hooks
        self.reasons = []
        self.detached = None

    def push(self, reason):
        self.reasons.append(reason)
        if self.detached is None:
            self.detached = _detach(self.hooks)

    def pop(self, reason):
        if reason in self.reasons:
            self.reasons.remove(reason)
        if not self.reasons and self.detached is not None:
            detached, self.detached = self.detached, None
            _reattach(detached)


def _active_adapters(model):
    try:
        active = model.active_adapter
    except Exception:
        return None
    if isinstance(active, str):
        return frozenset((active,))
    try:
        return frozenset(active)
    except TypeError:
        return None


def _router_extra_args(module, parent):
    # Routers such as MiniMax-M2 / HY-V3 (`forward(hidden_states, e_score_correction_bias)`) or
    # LFM2-MoE (`forward(hidden_states, expert_bias = None)`) take the parent MoE block's buffer:
    # the block calls `self.gate(hidden_states, self.e_score_correction_bias)`. Pass the parent's
    # same-named tensor; None when a required argument has none.
    try:
        params = list(inspect.signature(module.forward).parameters.values())[1:]
    except (TypeError, ValueError):
        return {}
    extra = {}
    for param in params:
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        value = getattr(parent, param.name, None) if parent is not None else None
        if torch.is_tensor(value):
            extra[param.name] = value
        elif param.default is param.empty:
            return None
    return extra


def _prime_formula(
    rr,
    module,
    hook,
    layers,
    parent = None,
):
    """Run the zoo hook once, eagerly, on a synthetic router call whose replay is the natural
    routing: it picks and caches module._unsloth_replay_formula exactly as the first real call
    would. Returns None when primed, else why it could not be."""
    weight = getattr(module, "weight", None)
    hidden = getattr(module, "hidden_dim", None) or getattr(module, "in_features", None)
    if hidden is None and torch.is_tensor(weight) and weight.dim() == 2:
        hidden = weight.shape[-1]
    if not isinstance(hidden, int) or not torch.is_tensor(weight) or not weight.is_floating_point():
        return "its hidden size or weight is unknown"
    extra = _router_extra_args(module, parent)
    if extra is None:
        return "its forward needs an argument the parent MoE block does not hold"
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(8, hidden, generator = generator).to(weight.device, weight.dtype)
    state = rr._STATE
    fields = ("current", "in_forward", "record", "pending", "recomputed", "changed")
    old = {f: getattr(state, f) for f in fields}
    old_stats = dict(state.stats)
    if getattr(module, "_unsloth_replay_formula", None) is not None:
        return _check_formula(rr, module, x, extra)  # already picked (an earlier install)
    try:
        with torch.no_grad():
            output = module.forward(x, **extra)  # no hooks: the zoo hook is called by hand below
            positions = rr._split_router_output(output)
            if positions is None:
                return "its outputs are not (logits, weights, ids)"
            ids = output[positions[2]].reshape(-1, output[positions[2]].shape[-1]).long()
            state.current = ids[:, None, :].expand(-1, layers, -1).contiguous()
            state.in_forward, state.record, state.pending, state.recomputed = True, None, [], False
            hook(module, (x, *extra.values()), output)
    except Exception as error:
        return f"priming raised {type(error).__name__}: {error}"
    finally:
        for f, v in old.items():
            setattr(state, f, v)
        state.stats.clear()
        state.stats.update(old_stats)
    if getattr(module, "_unsloth_replay_formula", None) is None:
        return "the routing replay hook did not pick a weight formula"
    return _check_formula(rr, module, x, extra)


# The check also runs with the input scaled so the router logits spread to these standard
# deviations: near-zero logits make every formula give ~1 / k, so the priming input alone can let a
# wrong formula through (sigmoid routing matched by softmax).
_CHECK_LOGIT_STDS = (1.0, 4.0)


def _check_formula(rr, module, x, extra):
    """None when the cached zoo formula reproduces this router's own top-k weights, else why not.
    The zoo stores False when it found none (sigmoid / correction bias routers, MiniMax-M2, HY-V3,
    LFM2-MoE): replay would silently keep natural routing there, so it counts as a failure too."""
    formula = getattr(module, "_unsloth_replay_formula", None)
    if formula is False or formula is None:
        return "routing replay has no weight formula for this router (e.g. sigmoid routing)"
    fn = rr._formulas().get(formula) if hasattr(rr, "_formulas") else None
    if fn is None:
        return f"unknown routing replay weight formula {formula!r}"
    with torch.no_grad():
        output = module.forward(x, **extra)
        positions = rr._split_router_output(output)
        if positions is None:
            return "its outputs are not (logits, weights, ids)"
        spread = float(output[positions[0]].float().std())
        scales = [1.0] + ([min(t / spread, 1e6) for t in _CHECK_LOGIT_STDS] if spread > 0 else [])
        for scale in scales:
            output = module.forward(x * scale, **extra)
            positions = rr._split_router_output(output)
            if positions is None:
                return "its outputs are not (logits, weights, ids)"
            li, wi, ii, dense = positions
            logits = output[li].reshape(-1, output[li].shape[-1])
            ids = output[ii].reshape(-1, output[ii].shape[-1]).long()
            weights = (
                output[wi].reshape(logits.shape).gather(-1, ids)
                if dense
                else output[wi].reshape(ids.shape)
            )
            candidate = fn(logits, ids).to(weights.dtype).float()
            if not torch.allclose(candidate, weights.float(), rtol = 2e-2, atol = 2e-3):
                return f"the {formula} weight formula does not reproduce its routing weights"
    return None


def _remove_hook(module, name, extras, hook_id):
    for table in [getattr(module, name, None)] + [getattr(module, e, None) for e in extras]:
        if table is not None:
            table.pop(hook_id, None)


def prepare_routing_replay(model, before):
    """After install_routing_replay(model, ...): prime the router formulas outside compilation and
    detach the replay hooks during `model.disable_adapter()` and while `model.set_adapter()` leaves a
    non-policy adapter active (PEFT reference scoring). A router that
    cannot be primed loses its replay hook (it keeps its natural routing) with a warning, rather
    than reaching the data dependent formula pick inside a compiled graph. Returns the number of
    replay hooks left (0 when routing replay installed nothing)."""
    hooks = _new_hooks(model, before)
    if not hooks:
        return 0
    try:
        import unsloth_zoo.routing_replay as rr
    except Exception:
        rr = None
    if rr is not None and all(
        hasattr(rr, n) for n in ("_STATE", "_is_router", "_split_router_output", "_LAYER_INDEX_RE")
    ):
        layer_of, parent_of = {}, {}
        for name, module in model.named_modules():
            match = rr._LAYER_INDEX_RE.search(f"{name}.")
            if match is not None:
                layer_of[id(module)] = int(match.group(1))
            for child in module.children():
                parent_of[id(child)] = module
        routers = [
            entry
            for entry in hooks
            if entry[1] == "_forward_hooks" and rr._is_router(entry[0]) and id(entry[0]) in layer_of
        ]
        # One replay row per decoder layer index, every row the natural routing.
        layers = 1 + max((layer_of[id(m)] for m, *_ in routers), default = 0)
        failed = []
        for entry in routers:
            module, name, extras, hook_id = entry
            why = _prime_formula(
                rr, module, module._forward_hooks[hook_id], layers, parent_of.get(id(module))
            )
            if why is not None:
                _remove_hook(*entry)
                failed.append((type(module).__name__, why))
        if failed:
            removed = {id(e) for e in routers if e[3] not in getattr(e[0], e[1], {})}
            hooks = [e for e in hooks if id(e) not in removed]
            names = sorted({f"{cls} ({why})" for cls, why in failed})
            warnings.warn(
                f"Unsloth: routing replay is off for {len(failed)} router(s) it could not prepare, "
                f"which keep their natural routing: {'; '.join(names)}.",
                stacklevel = 2,
            )

    detacher = getattr(model, "_unsloth_routing_replay_detacher", None)
    if detacher is None:
        detacher = _ReplayDetacher(hooks)
        try:
            model._unsloth_routing_replay_detacher = detacher
        except Exception:
            pass

    disable_adapter = getattr(model, "disable_adapter", None)
    if callable(disable_adapter) and not getattr(disable_adapter, "_unsloth_routing_replay", False):

        @contextlib.contextmanager
        def disable_adapter_without_replay(*args, **kwargs):
            with disable_adapter(*args, **kwargs):
                detacher.push("disable_adapter")
                try:
                    yield
                finally:
                    detacher.pop("disable_adapter")

        disable_adapter_without_replay._unsloth_routing_replay = True
        model.disable_adapter = disable_adapter_without_replay

    # TRL use_adapter(model, "ref") calls set_adapter: score it with natural routing like disable_adapter().
    policy = _active_adapters(model)
    set_adapter = getattr(model, "set_adapter", None)
    if (
        policy
        and callable(set_adapter)
        and not getattr(set_adapter, "_unsloth_routing_replay", False)
    ):

        def set_adapter_without_replay(*args, **kwargs):
            out = set_adapter(*args, **kwargs)
            active = _active_adapters(model)
            if active is not None and active != policy:
                if "set_adapter" not in detacher.reasons:
                    detacher.push("set_adapter")
            else:
                detacher.pop("set_adapter")
            return out

        set_adapter_without_replay._unsloth_routing_replay = True
        model.set_adapter = set_adapter_without_replay
    return len(hooks)
