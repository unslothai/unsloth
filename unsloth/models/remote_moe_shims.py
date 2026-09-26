# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Train-mode forwards for DeepSeek-V3-derived remote MoE ports (Kimi-K2.7, sarvam) whose gate
asserts `not self.training` and whose block only computes under no-grad `moe_infer`. Keyed on
structure, never repo name; no-op for transformers' own models."""

import functools

import torch

__all__ = [
    "prepare_remote_moe_for_training",
    "is_remote_deepseek_gate",
    "is_remote_deepseek_moe",
    "packed_expert_target_parameters",
]


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
    import ast

    def value_in_training(node):
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
        test = branch.test
        if not any(isinstance(n, ast.Attribute) and n.attr == "training" for n in ast.walk(test)):
            return []
        value = value_in_training(test)
        if value is None:
            return [branch.body, branch.orelse]
        return [branch.body] if value else [branch.orelse]

    def only_raises(branch):
        return all(
            isinstance(stmt, (ast.Raise, ast.Pass))
            or (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))
            for stmt in branch
        )

    for node in body:
        for sub in ast.walk(node):
            if not isinstance(sub, ast.If):
                continue
            for branch in training_branches(sub):
                if branch and not only_raises(branch) and not _calls_moe_infer(branch):
                    return True
    return False


def _forward_has_no_training_branch(cls) -> bool:
    """True for inference-only ports (training reaches `moe_infer` or nothing); unreadable source -> False."""
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
    # Structural match: derived code renames the block (sarvam's `SarvamMLAMoE`).
    cls = type(module)
    experts = getattr(module, "experts", None)
    return (
        _is_remote_code(cls)
        and (isinstance(experts, torch.nn.ModuleList) or _is_packed_experts(experts))
        and hasattr(module, "moe_infer")
        and hasattr(module, "gate")
        and _forward_has_no_training_branch(cls)
    )


def _gate_forward_without_training_assert(original):
    @functools.wraps(original)
    def forward(self, hidden_states):
        if not self.training:
            return original(self, hidden_states)
        # Gate has no children: flipping `training` only bypasses the assert.
        self.training = False
        try:
            return original(self, hidden_states)
        finally:
            self.training = True

    forward._unsloth_remote_moe_shim = True
    return forward


_MISSING = object()


def _is_packed_experts(experts) -> bool:
    while hasattr(experts, "base_layer"):
        experts = experts.base_layer
    return getattr(type(experts), "_unsloth_mxfp4_stacked_experts", False) is True


def _packed_moe_dispatch(block, x, topk_idx, topk_weight):
    # Through the experts module so a PEFT expert LoRA wrapper still applies.
    return block.experts(x, topk_idx, topk_weight)


def _moe_train_dispatch(block, x, topk_idx, topk_weight):
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
    # Reuse the port's forward (eval flags, swapped moe_infer): rebuilding it would skip Kimi-K3's latent projections.
    @functools.wraps(original)
    def forward(self, hidden_states):
        if getattr(self, "ep_size", 1) > 1:
            return original(self, hidden_states)
        packed = _is_packed_experts(getattr(self, "experts", None))
        if not self.training and not packed:
            return original(self, hidden_states)
        was_training = self.training
        gate = getattr(self, "gate", None)
        gate_was_training = isinstance(gate, torch.nn.Module) and gate.training
        own = self.__dict__.get("moe_infer", _MISSING)
        # Plain attribute flip: children keep their train-mode flags.
        self.training = False
        if gate_was_training:
            gate.training = False
        dispatch = _packed_moe_dispatch if packed else _moe_train_dispatch
        self.__dict__["moe_infer"] = functools.partial(dispatch, self)
        try:
            return original(self, hidden_states)
        finally:
            self.training = was_training
            if gate_was_training:
                gate.training = True
            if own is _MISSING:
                self.__dict__.pop("moe_infer", None)
            else:
                self.__dict__["moe_infer"] = own

    forward._unsloth_remote_moe_shim = True
    return forward


def _rebind_accelerate_hook(module):
    # device_map hooks keep the bound original in `_old_forward`, bypassing class patches (multi-GPU Kimi hit the assert).
    if "_old_forward" not in vars(module):
        return False
    import types

    module._old_forward = types.MethodType(type(module).forward, module)
    return True


def prepare_remote_moe_for_training(model, verbose = True):
    """Idempotent; returns the names of the patched classes."""
    patched = []
    seen = set()
    shimmed_classes = set()
    # Only gates of inference-only blocks: eval mode would drop a trainable gate's aux loss.
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


def _names_a_packed_expert(entry, stack, leaf):
    parts = entry.split(".")
    template = stack.split(".") + [None, leaf]
    if len(parts) > len(template):
        return False
    return all(
        part.isdigit() if want is None else part == want
        for part, want in zip(parts, template[len(template) - len(parts) :])
    )


def packed_expert_target_parameters(model, target_parameters, requested_leaves):
    """Packed MXFP4 expert LoRA stays opt in: keep a stack's gate_up_proj (w1/w3) / down_proj (w2) only if
    the request named its pre-stacking expert Linears, matched as PEFT would (regex fullmatch / dotted suffix)."""
    counts = {}
    for name, m in model.named_modules():
        if _is_packed_experts(m):
            while hasattr(m, "base_layer"):
                m = m.base_layer
            counts[name] = max(int(getattr(m, "num_experts", 1) or 1), 1)
    stacks = list(counts)
    if not stacks:
        return target_parameters
    names = ("experts.gate_up_proj", "experts.down_proj")
    kept = [p for p in (target_parameters or []) if not p.endswith(names)]
    if isinstance(requested_leaves, str):
        import re
        def named(stack, leaf):
            return any(
                re.fullmatch(requested_leaves, f"{stack}.{e}.{leaf}") for e in range(counts[stack])
            )
    else:

        def named(stack, leaf):
            return any(
                _names_a_packed_expert(str(entry), stack, leaf) for entry in requested_leaves or ()
            )

    for leaves, projection in ((("w1", "w3"), "gate_up_proj"), (("w2",), "down_proj")):
        chosen = [stack for stack in stacks if any(named(stack, leaf) for leaf in leaves)]
        # PEFT suffix-matches target_parameters, so a request scoped to some layers stays scoped.
        if len(chosen) == len(stacks):
            kept.append(f"experts.{projection}")
        else:
            kept.extend(f"{stack}.{projection}" for stack in chosen)
    return kept or None
