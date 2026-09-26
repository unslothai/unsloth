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

    for node in body:
        for sub in ast.walk(node):
            if not isinstance(sub, ast.If):
                continue
            for branch in training_branches(sub):
                if branch and not _calls_moe_infer(branch):
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
        # Gate has no children: flipping `training` only bypasses the assert.
        self.training = False
        try:
            return original(self, hidden_states)
        finally:
            self.training = True

    forward._unsloth_remote_moe_shim = True
    return forward


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
    @functools.wraps(original)
    def forward(self, hidden_states):
        if not self.training or getattr(self, "ep_size", 1) > 1:
            return original(self, hidden_states)
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        y = _moe_train_dispatch(self, flat, topk_idx, topk_weight).view(*orig_shape)
        # Config key varies (sarvam: `num_shared_experts`); the module decides.
        shared = getattr(self, "shared_experts", None)
        if shared is not None:
            y = y + shared(identity)
        return y

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
