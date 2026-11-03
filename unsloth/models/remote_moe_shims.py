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
"""Train-mode forwards for DeepSeek-V3-derived remote MoE ports, keyed on structure never repo name;
no-op for the native transformers models."""

import functools
import os
import threading

import torch

__all__ = [
    "prepare_remote_moe_for_training",
    "is_remote_deepseek_gate",
    "is_remote_deepseek_moe",
    "packed_expert_target_parameters",
    "prepare_remote_moe_for_inference",
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


# Eval routing without a host sync. The remote `moe_infer` reads per-expert token counts back to the host and
# loops over experts, so each MoE layer syncs and launches one GEMM chain per active expert. The grouped version
# keeps the counts on the device and runs every expert in three grouped GEMMs; every routed token is computed.


_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_STACK_LOCK = threading.Lock()


def _plain_expert(expert, dtype):
    names = _EXPERT_PROJECTIONS
    return all(
        type(getattr(expert, name, None)) is torch.nn.Linear
        and getattr(expert, name).bias is None
        and getattr(expert, name).weight.dtype == dtype
        for name in names
    ) and hasattr(expert, "act_fn")


def _torch_grouped_mm(inputs, weight, offsets):
    return torch._grouped_mm(inputs, weight, offs = offsets)


def _triton_grouped_mm(inputs, weight, offsets):
    from unsloth_zoo.temporary_patches.moe_grouped_fp16 import generic_grouped_mm
    return generic_grouped_mm(inputs, weight, offsets)


_FALLBACK_LOGGED = set()


def _log_fallback(reason):
    if reason not in _FALLBACK_LOGGED:
        _FALLBACK_LOGGED.add(reason)
        print(
            f"Unsloth: MoE eval routing keeps the per-expert loop ({reason}); generation is slower."
        )


def _grouped_mm_candidates(weight):
    """Grouped GEMMs that keep the offsets on the device, best first. torch._grouped_mm only has a real kernel on
    sm90 / sm100 (and ROCm): elsewhere torch 2.9+ loops over host-copied offsets (a sync per layer that also
    breaks CUDA Graph capture) and 2.8 raises, so sm80 / sm89 / sm120 use unsloth_zoo's Triton grouped GEMM.
    UNSLOTH_MOE_INFER_GROUPED=torch|triton|0 forces one or turns the grouped path off."""
    choice = os.environ.get("UNSLOTH_MOE_INFER_GROUPED", "auto").strip().lower()
    if choice in ("0", "false", "off"):
        return []
    candidates = []
    if hasattr(torch, "_grouped_mm") and choice in ("auto", "torch"):
        native = torch.version.hip is not None or torch.cuda.get_device_capability(weight.device)[
            0
        ] in (9, 10)
        if native or choice == "torch":
            candidates.append(_torch_grouped_mm)
    if choice in ("auto", "triton") and torch.version.hip is None:
        if torch.cuda.get_device_capability(weight.device) >= (8, 0):
            candidates.append(_triton_grouped_mm)
    return candidates


def _grouped_mm_for(weights):
    """The first grouped GEMM that reproduces a plain matmul on these weights' device / dtype / layout (a tiny
    call each), or None, logged once, so a refusal keeps the original loop instead of failing generation."""
    first = weights[0]
    if not first.is_cuda:
        return None
    if first.dtype not in (torch.bfloat16, torch.float16):
        _log_fallback(f"expert weights are {first.dtype}, not bfloat16 / float16")
        return None
    errors = []
    for grouped_mm in _grouped_mm_candidates(first):
        try:
            offsets = torch.tensor([1, 2], dtype = torch.int32, device = first.device)
            for weight in weights:
                weight = weight.detach()
                # Deterministic inputs: the global RNG stream a seeded sample relies on stays untouched.
                inputs = torch.arange(2 * weight.shape[1], device = weight.device).view(2, -1)
                inputs = inputs.float().sin().div(weight.shape[1] ** 0.5).to(weight.dtype)
                stacked = torch.stack([weight, weight.flip(0)])
                out = grouped_mm(inputs, stacked.transpose(-2, -1), offsets)
                reference = torch.cat([inputs[:1] @ weight.T, inputs[1:] @ weight.flip(0).T])
                if not torch.allclose(out.float(), reference.float(), rtol = 2e-2, atol = 2e-2):
                    raise RuntimeError("mismatch against a plain matmul")
            return grouped_mm
        except Exception as error:
            errors.append(
                f"{grouped_mm.__name__.strip('_')}: {type(error).__name__}: {error}"[:200]
            )
    _log_fallback("; ".join(errors) or "no grouped GEMM for this GPU")
    return None


def _grouped_experts_live(
    block,
    stacked,
    every = True,
):
    """The experts are still plain linears whose weights alias `stacked`. A LoRA wrap, `model.to(...)` or a
    reload breaks the alias; `every = False` checks only the first expert (cheap, per forward)."""
    experts = block.experts
    if len(experts) != stacked[0].shape[0]:
        return False
    for buffer, name in zip(stacked, _EXPERT_PROJECTIONS):
        for index in range(len(experts) if every else 1):
            linear = getattr(experts[index], name, None)
            if (
                type(linear) is not torch.nn.Linear
                or linear.weight.dtype != buffer.dtype
                or linear.weight.data_ptr() != buffer[index].data_ptr()
            ):
                return False
    return True


def _grouped_experts(block):
    """Stack the expert weights once into [E, out, in] buffers and point each expert's weights at its slice,
    so no weight is held twice. The Parameter objects stay the same (an optimizer built earlier still owns
    them). None when the experts are not plain bf16 / fp16 linears on a device torch._grouped_mm serves."""
    cached = block.__dict__.get("_unsloth_grouped_experts", _MISSING)
    if cached is not _MISSING and (
        cached is None or _grouped_experts_live(block, cached, every = False)
    ):
        return cached
    # One thread stacks and re-points the weights; a concurrent first generate waits instead of capturing a
    # graph over half re-pointed experts.
    with _STACK_LOCK:
        cached = block.__dict__.get("_unsloth_grouped_experts", _MISSING)
        if cached is not _MISSING and (
            cached is None or _grouped_experts_live(block, cached, every = False)
        ):
            return cached
        return _stack_experts(block)


def _stack_experts(block):
    from .diffusion_infer import drop_on_move

    experts = block.experts
    stacked = grouped_mm = None
    first = experts[0] if isinstance(experts, torch.nn.ModuleList) and len(experts) else None
    if (
        first is not None
        and isinstance(getattr(first, "gate_proj", None), torch.nn.Linear)
        and all(_plain_expert(expert, first.gate_proj.weight.dtype) for expert in experts)
    ):
        grouped_mm = _grouped_mm_for((first.gate_proj.weight, first.down_proj.weight))
    elif first is not None and any(p.is_cuda for p in first.parameters()):
        _log_fallback("experts are not plain bias-free nn.Linear layers (quantized or adapters)")
    if grouped_mm is not None:
        stacked = []
        for name in _EXPERT_PROJECTIONS:
            buffer = torch.stack([getattr(expert, name).weight.detach() for expert in experts])
            for index, expert in enumerate(experts):
                getattr(expert, name).weight.data = buffer[index]
            stacked.append(buffer)
        stacked = (*stacked, grouped_mm)
        # model.to("cpu") moves the experts' parameters but not this tuple: drop it so the GPU stack is freed.
        drop_on_move(block, "_unsloth_grouped_experts")
    block.__dict__["_unsloth_grouped_experts"] = stacked
    return stacked


def _grouped_moe_infer(original):
    @functools.wraps(original)
    def moe_infer(self, x, topk_ids, topk_weight):
        stacked = None
        if not self.training and not torch.is_grad_enabled():
            stacked = _grouped_experts(self)
        if stacked is None or x.dtype != stacked[0].dtype:
            return original(self, x, topk_ids, topk_weight)
        gate, up, down, grouped_mm = stacked
        num_experts = gate.shape[0]
        counts = topk_ids.new_zeros((topk_ids.shape[0], num_experts))
        counts.scatter_(1, topk_ids, 1)
        offsets = counts.sum(dim = 0).cumsum(0).to(torch.int32)
        order = topk_ids.view(-1).argsort()
        sorted_tokens = x[order // topk_ids.shape[1]]
        act = self.experts[0].act_fn
        hidden = act(grouped_mm(sorted_tokens, gate.transpose(-2, -1), offsets))
        hidden = hidden * grouped_mm(sorted_tokens, up.transpose(-2, -1), offsets)
        outs = grouped_mm(hidden, down.transpose(-2, -1), offsets)
        new_x = torch.empty_like(outs)
        new_x[order] = outs
        return (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim = -1))
            .sum(dim = 1)
            .type(new_x.dtype)
        )

    moe_infer._unsloth_grouped_moe_infer = True
    return moe_infer


def _is_host_synced_moe_infer(cls) -> bool:
    import inspect

    infer = cls.__dict__.get("moe_infer")
    if infer is None or getattr(infer, "_unsloth_grouped_moe_infer", False):
        return False
    try:
        source = inspect.getsource(infer)
    except (OSError, TypeError):
        return False
    # The DeepSeek-style loop: counts to the host, then one expert at a time.
    return ".cpu()" in source and "self.experts[" in source


def prepare_remote_moe_for_inference(model):
    """Patch remote MoE blocks whose eval `moe_infer` syncs the host per layer (LLaDA2) to the grouped dispatch.
    Training, LoRA-wrapped or quantized experts and CPU tensors keep the original path."""
    patched = []
    for module in model.modules():
        cls = type(module)
        if (
            _is_remote_code(cls)
            and isinstance(getattr(module, "experts", None), torch.nn.ModuleList)
            and _is_host_synced_moe_infer(cls)
        ):
            cls.moe_infer = _grouped_moe_infer(cls.__dict__["moe_infer"])
            patched.append(cls.__name__)
        # Re-decide stacks that went stale (LoRA added, model moved or reloaded, dtype changed) or were
        # refused earlier (e.g. first seen on the CPU), so a stale buffer is neither used nor kept alive.
        cached = module.__dict__.get("_unsloth_grouped_experts", _MISSING)
        if cached is not _MISSING and (cached is None or not _grouped_experts_live(module, cached)):
            module.__dict__.pop("_unsloth_grouped_experts")
    return patched
