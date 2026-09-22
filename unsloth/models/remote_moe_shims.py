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
"""Training forwards for remote MoE code that only ships an inference path.

The DeepSeek-V3 modeling file that Kimi-K2.7 (and every other DeepSeek-V3 derived
`trust_remote_code` checkpoint) carries is an inference port: `MoEGate.forward`
asserts `not self.training` in front of its no-aux-loss routing, and
`DeepseekV3MoE.forward` computes the expert mix only under `if not self.training`,
through a `@torch.no_grad()` `moe_infer`. In train mode the block raises, and with
the assert removed it would return an unbound `y`.

`prepare_remote_moe_for_training(model)` repairs both on the checkpoint's dynamically
created classes and nowhere else:

* the gate runs its own routing code with the training assert bypassed. The maths is
  unchanged and already differentiable (the selected weights are gathered from the
  sigmoid scores), so the router trains the way transformers' native DeepSeek-V3
  router does;
* the MoE block gets a training path that dispatches tokens to their experts and
  sums the weighted outputs with autograd on, the same computation as `moe_infer`
  without `no_grad` and without the expert-parallel branch. Eval mode still takes the
  checkpoint's own `moe_infer`.

Everything is keyed on structure (class names plus the attributes the port defines),
never on the repo name, and is a no-op for transformers' own models.
"""

import functools

import torch

__all__ = ["prepare_remote_moe_for_training", "is_remote_deepseek_gate", "is_remote_deepseek_moe"]


def _is_remote_code(cls) -> bool:
    return "transformers_modules" in (getattr(cls, "__module__", "") or "")


def is_remote_deepseek_gate(module) -> bool:
    cls = type(module)
    return (
        cls.__name__ == "MoEGate"
        and _is_remote_code(cls)
        and all(
            hasattr(module, a)
            for a in (
                "topk_method",
                "n_group",
                "topk_group",
                "routed_scaling_factor",
                "norm_topk_prob",
                "top_k",
            )
        )
    )


def _calls_moe_infer(nodes) -> bool:
    import ast
    return any(
        isinstance(sub, ast.Attribute) and sub.attr == "moe_infer"
        for node in nodes
        for sub in ast.walk(node)
    )


def _has_own_training_dispatch(body) -> bool:
    """Whether a `self.training` branch computes the output itself: non-empty and not
    through the no-grad `moe_infer`."""
    import ast

    def value_in_training(node):
        """The predicate's value with self.training True: True, False or None (unknown)."""
        if isinstance(node, ast.Attribute) and node.attr == "training":
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            inner = value_in_training(node.operand)
            return None if inner is None else not inner
        if isinstance(node, ast.BoolOp):
            values = [value_in_training(value) for value in node.values]
            decisive = isinstance(node.op, ast.Or)
            if decisive in values:
                return decisive
            if all(value is (not decisive) for value in values):
                return not decisive
        return None

    def training_branches(branch):
        """The branches of this `if` that training can reach, when its test reads self.training
        (at any depth of and / or / not)."""
        test = branch.test
        if not any(isinstance(n, ast.Attribute) and n.attr == "training" for n in ast.walk(test)):
            return []
        value = value_in_training(test)
        if value is None:
            return [branch.body, branch.orelse]
        return [branch.body] if value else [branch.orelse]

    for node in body:
        for sub in ast.walk(node):
            if not isinstance(sub, ast.If):
                continue
            for branch in training_branches(sub):
                if branch and not _calls_moe_infer(branch):
                    return True
    return False


def _forward_has_no_training_branch(cls) -> bool:
    """Inference-only ports reach the no-grad `moe_infer` in training or leave training
    empty (sarvam's `else:` calls it too; Kimi's port has no `else:`); training-capable
    DeepSeek-V2/V3 code dispatches its own experts under `self.training`, in either branch
    order. Only the former is shimmed. A forward whose source cannot be read is left alone."""
    import ast
    import inspect
    import textwrap

    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls.forward)))
    except (OSError, TypeError, SyntaxError):
        return False
    function = tree.body[0] if tree.body else None
    if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return _calls_moe_infer(function.body) and not _has_own_training_dispatch(function.body)


def is_remote_deepseek_moe(module) -> bool:
    # Matched on structure rather than class name: DeepSeek-derived remote code renames the
    # block (sarvam's `SarvamMLAMoE`) but keeps the same inference-only port.
    cls = type(module)
    return (
        _is_remote_code(cls)
        and isinstance(getattr(module, "experts", None), torch.nn.ModuleList)
        and hasattr(module, "moe_infer")
        and hasattr(module, "gate")
        and _forward_has_no_training_branch(cls)
    )


def _gate_forward_without_training_assert(original):
    @functools.wraps(original)
    def forward(self, hidden_states):
        if not self.training:
            return original(self, hidden_states)
        # `training` is a plain attribute on nn.Module; the gate has no children, so
        # flipping it for the call changes nothing but the assert.
        self.training = False
        try:
            return original(self, hidden_states)
        finally:
            self.training = True

    forward._unsloth_remote_moe_shim = True
    return forward


_MISSING = object()


def _moe_train_dispatch(block, x, topk_idx, topk_weight):
    """Differentiable version of the port's `moe_infer` for a single rank."""
    num_tokens, top_k = topk_idx.shape
    experts = block.experts
    flat_idx = topk_idx.reshape(-1)
    order = torch.argsort(flat_idx, stable = True)
    sorted_tokens = x[order // top_k]
    counts = torch.bincount(flat_idx, minlength = len(experts)).tolist()
    outputs = []
    start = 0
    for expert_index, count in enumerate(counts):
        if count == 0:
            continue
        expert = experts[expert_index]
        outputs.append(expert(sorted_tokens[start : start + count]))
        start += count
    if outputs:
        outs = torch.cat(outputs, dim = 0)
    else:
        outs = sorted_tokens.new_zeros((0, x.shape[-1]))
    unsorted = torch.empty_like(outs)
    unsorted[order] = outs
    weighted = unsorted.view(num_tokens, top_k, -1).to(torch.float32) * topk_weight.to(
        torch.float32
    ).unsqueeze(-1)
    return weighted.sum(dim = 1).to(x.dtype)


def _moe_forward_with_training_path(original):
    """Run the port's own forward in train mode, with the block and its gate reporting
    eval and `moe_infer` swapped for the differentiable dispatch on this instance only.

    Everything the port does around the expert mix is kept as written: DeepSeek adds the
    shared experts, Kimi-K3's latent MoE wraps the mix in `routed_expert_down_proj`, an
    RMSNorm and `routed_expert_up_proj`. Rebuilding the forward instead would feed the
    experts the full hidden size and skip those projections."""

    @functools.wraps(original)
    def forward(self, hidden_states):
        if not self.training or getattr(self, "ep_size", 1) > 1:
            return original(self, hidden_states)
        gate = getattr(self, "gate", None)
        gate_was_training = isinstance(gate, torch.nn.Module) and gate.training
        own = self.__dict__.get("moe_infer", _MISSING)
        # `training` is a plain attribute; flipping it on the block and its gate changes
        # only their own `if not self.training` / `assert not self.training` checks, the
        # experts and projections below keep their train-mode flags.
        self.training = False
        if gate_was_training:
            gate.training = False
        self.__dict__["moe_infer"] = functools.partial(_moe_train_dispatch, self)
        try:
            return original(self, hidden_states)
        finally:
            self.training = True
            if gate_was_training:
                gate.training = True
            if own is _MISSING:
                self.__dict__.pop("moe_infer", None)
            else:
                self.__dict__["moe_infer"] = own

    forward._unsloth_remote_moe_shim = True
    return forward


def _rebind_accelerate_hook(module):
    """Point an accelerate hook attached during loading at the shimmed forward.

    A `device_map` load wraps every dispatched module's `forward` before the shims run and
    keeps the bound original as `module._old_forward`, so a class-level patch is never reached
    from `module(...)` on a multi-GPU model: Kimi-K2.7-Code on four cards still hit the gate's
    `assert not self.training`. Single-GPU loads have no hook and nothing to rebind."""
    if "_old_forward" not in vars(module):
        return False
    import types

    module._old_forward = types.MethodType(type(module).forward, module)
    return True


def prepare_remote_moe_for_training(model, verbose = True):
    """Patch the remote DeepSeek-style gate and MoE classes found in `model` so the block
    runs in train mode. Idempotent; returns the names of the classes it patched."""
    patched = []
    seen = set()
    shimmed_classes = set()
    # A gate is patched only when the MoE block that owns it is the inference-only port: a
    # training-capable block's gate computes its auxiliary routing loss in train mode, and
    # running it in eval mode would silently drop that loss.
    shimmable_gates = set()
    for module in model.modules():
        gate = getattr(module, "gate", None)
        if gate is not None and is_remote_deepseek_moe(module):
            shimmable_gates.add(type(gate))
    for module in model.modules():
        cls = type(module)
        if cls in seen:
            if cls in shimmed_classes:
                _rebind_accelerate_hook(module)
            continue
        seen.add(cls)
        current = cls.__dict__.get("forward")
        if getattr(current, "_unsloth_remote_moe_shim", False):
            # Already patched in an earlier call; a hook attached since still needs rebinding.
            shimmed_classes.add(cls)
            _rebind_accelerate_hook(module)
            continue
        if is_remote_deepseek_gate(module) and cls in shimmable_gates:
            cls.forward = _gate_forward_without_training_assert(cls.forward)
            patched.append(cls.__name__)
            shimmed_classes.add(cls)
            _rebind_accelerate_hook(module)
        elif is_remote_deepseek_moe(module):
            cls.forward = _moe_forward_with_training_path(cls.forward)
            patched.append(cls.__name__)
            shimmed_classes.add(cls)
            _rebind_accelerate_hook(module)
    if patched and verbose:
        print(
            "Unsloth: The remote MoE code only had an inference path; added a training forward to "
            + ", ".join(patched)
            + "."
        )
    return patched
