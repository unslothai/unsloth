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


def _forward_has_no_training_branch(cls) -> bool:
    """The inference-only ports compute the output only under `if not self.training` and
    leave nothing for training; the training-capable DeepSeek-V2/V3 implementations have an
    `if self.training:` dispatch of their own (and a gate that returns an auxiliary loss).
    Only the former is shimmed. A forward whose source cannot be read is left alone."""
    import inspect

    try:
        source = inspect.getsource(cls.forward)
    except (OSError, TypeError):
        return False
    # The port: `if not self.training: y = self.moe_infer(...)` and nothing for training.
    # The training-capable implementation: `if self.training: ... else: ... moe_infer(...)`.
    return "moe_infer" in source and "if self.training" not in source


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
    @functools.wraps(original)
    def forward(self, hidden_states):
        if not self.training or getattr(self, "ep_size", 1) > 1:
            return original(self, hidden_states)
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        flat = hidden_states.view(-1, hidden_states.shape[-1])
        y = _moe_train_dispatch(self, flat, topk_idx, topk_weight).view(*orig_shape)
        # DeepSeek ports only create `shared_experts` when `n_shared_experts` is set, and
        # others (sarvam: `num_shared_experts`) store None, so the module itself decides.
        shared = getattr(self, "shared_experts", None)
        if shared is not None:
            y = y + shared(identity)
        return y

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
