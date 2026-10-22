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
- PEFT reference scoring (`disable_adapter()`) must see the base model's own routing, as a separate
  reference model does, so the hooks are detached inside that context.
"""

import contextlib

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


@contextlib.contextmanager
def _detached(hooks):
    # Hook order is call order, and ids do not give it (register_*_hook(prepend = True) puts a newer
    # id first): keep each table's own key order and put it back exactly.
    orders = {}
    removed = []
    for module, name, extras, hook_id in hooks:
        for table in [getattr(module, name, None)] + [getattr(module, e, None) for e in extras]:
            if table is None or hook_id not in table:
                continue
            orders.setdefault(id(table), (table, list(table.keys())))
            removed.append((table, hook_id, table.pop(hook_id)))
    try:
        yield
    finally:
        for table, hook_id, value in removed:
            table[hook_id] = value
        for table, order in orders.values():
            known = set(order)
            added = [k for k in table if k not in known]  # registered inside the context: keep last
            items = dict(table)
            table.clear()
            for k in order + added:
                if k in items:
                    table[k] = items[k]


def _prime_formula(rr, module, hook, layers):
    # Run the zoo hook once, eagerly, on a synthetic router call whose replay is the natural routing:
    # it picks and caches module._unsloth_replay_formula exactly as the first real call would.
    if getattr(module, "_unsloth_replay_formula", None) is not None:
        return
    weight = getattr(module, "weight", None)
    hidden = getattr(module, "hidden_dim", None) or getattr(module, "in_features", None)
    if hidden is None and torch.is_tensor(weight) and weight.dim() == 2:
        hidden = weight.shape[-1]
    if not isinstance(hidden, int) or not torch.is_tensor(weight) or not weight.is_floating_point():
        return
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(8, hidden, generator = generator).to(weight.device, weight.dtype)
    state = rr._STATE
    fields = ("current", "in_forward", "record", "pending", "recomputed", "changed")
    old = {f: getattr(state, f) for f in fields}
    old_stats = dict(state.stats)
    try:
        with torch.no_grad():
            output = module.forward(x)  # no hooks: the zoo hook is called by hand below
            positions = rr._split_router_output(output)
            if positions is None:
                return
            ids = output[positions[2]].reshape(-1, output[positions[2]].shape[-1]).long()
            state.current = ids[:, None, :].expand(-1, layers, -1).contiguous()
            state.in_forward, state.record, state.pending, state.recomputed = True, None, [], False
            hook(module, (x,), output)
    finally:
        for f, v in old.items():
            setattr(state, f, v)
        state.stats.clear()
        state.stats.update(old_stats)


def prepare_routing_replay(model, before):
    """After install_routing_replay(model, ...): prime the router formulas outside compilation and
    detach the replay hooks during `model.disable_adapter()` (PEFT reference scoring). Returns the
    number of hooks found (0 when routing replay installed nothing)."""
    hooks = _new_hooks(model, before)
    if not hooks:
        return 0
    try:
        import unsloth_zoo.routing_replay as rr
    except Exception:
        rr = None
    if rr is not None and all(hasattr(rr, n) for n in ("_STATE", "_is_router", "_split_router_output", "_LAYER_INDEX_RE")):
        layer_of = {}
        for name, module in model.named_modules():
            match = rr._LAYER_INDEX_RE.search(f"{name}.")
            if match is not None:
                layer_of[id(module)] = int(match.group(1))
        routers = [
            (m, m._forward_hooks[h]) for m, name, _, h in hooks
            if name == "_forward_hooks" and rr._is_router(m) and id(m) in layer_of
        ]
        # One replay row per decoder layer index, every row the natural routing.
        layers = 1 + max((layer_of[id(m)] for m, _ in routers), default = 0)
        for module, hook in routers:
            try:
                _prime_formula(rr, module, hook, layers)
            except Exception:
                pass  # the first real call picks it then, as before

    disable_adapter = getattr(model, "disable_adapter", None)
    if callable(disable_adapter) and not getattr(disable_adapter, "_unsloth_routing_replay", False):

        @contextlib.contextmanager
        def disable_adapter_without_replay(*args, **kwargs):
            with disable_adapter(*args, **kwargs), _detached(hooks):
                yield

        disable_adapter_without_replay._unsloth_routing_replay = True
        model.disable_adapter = disable_adapter_without_replay
    return len(hooks)
