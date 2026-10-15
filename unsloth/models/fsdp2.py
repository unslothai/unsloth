# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""What Unsloth adds to accelerate's FSDP2 wrap.

accelerate (`fsdp2_prepare_model`) already shards one unit per `_no_split_modules` block, keeps
the [final norm, lm_head] tail gathered from forward to backward, groups tied embeddings, and
(>= 1.11) turns `no_sync` into `set_requires_gradient_sync(False)`. Unsloth fills in around it:

1. Mixed precision: accelerate's automatic policy reduces gradients in the compute dtype and casts
   the model outputs (the loss included) to it. Unsloth keeps the compute dtype for the gathered
   weights but reduces in float32 and leaves outputs alone.
2. The last decoder block keeps its weights gathered after forward (backward starts with it).
3. Unsloth's forwards read `lm_head.weight` and the final norm's weight directly (fused CE loss,
   fast RMSNorm) instead of calling those modules, so FSDP's pre-forward hook would never gather
   their unit: the unit is gathered at the start of the model's forward instead.
4. The fused LoRA kernels read `.weight` of q/k/v/o/gate/up/down directly. They run inside the
   decoder block's forward, after FSDP gathered that block, so they stay on when each block is its
   own unit and no submodule of a fused layer is a unit of its own. Otherwise they are declined.
5. `no_sync` on accelerate < 1.11 ignores FSDP2; it is mapped to `set_requires_gradient_sync`.

`UNSLOTH_FSDP2_ACCELERATE_DEFAULTS=1` skips 1 and 2.
"""

import contextlib
import dataclasses
import functools
import inspect
import os

import torch

__all__ = [
    "fsdp2_trainer_setup",
    "install_fsdp2_prepare_hook",
    "install_fsdp2_no_sync",
]

_PATCHED = "_unsloth_fsdp2_patched"
_FALSY = ("0", "false", "no", "off")
_TRUTHY = ("1", "true", "yes", "on")
# Unsloth's backbone forwards call fast_rms_layernorm on these instead of the module.
_DIRECT_READ_NORMS = ("norm", "final_layernorm")


def _fsdp_module_class():
    try:
        from torch.distributed.fsdp import FSDPModule
        return FSDPModule
    except ImportError:
        pass
    try:
        from torch.distributed._composable.fsdp import FSDPModule
        return FSDPModule
    except ImportError:
        return None


def _rank0():
    try:
        import torch.distributed as dist
        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True


def _print_rank0(message):
    if _rank0():
        print(message)


def _keep_accelerate_defaults():
    return os.environ.get("UNSLOTH_FSDP2_ACCELERATE_DEFAULTS", "0").strip().lower() in _TRUTHY


def _compute_dtype(accelerator):
    return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        str(getattr(accelerator, "mixed_precision", "no"))
    )


def fill_fsdp2_mixed_precision(accelerator):
    """Reduce gradients in float32 and stop casting the model outputs, only when the policy is
    the one accelerate derived from `mixed_precision` (every dtype equal to the compute dtype).
    A policy the user wrote is left alone. Returns the policy in force."""
    plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None)
    policy = getattr(plugin, "mixed_precision_policy", None)
    if policy is None or _keep_accelerate_defaults():
        return policy
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy
    except ImportError:
        return policy
    compute = _compute_dtype(accelerator)
    if compute is None or not isinstance(policy, MixedPrecisionPolicy):
        return policy
    if (
        policy.param_dtype != compute
        or policy.reduce_dtype != compute
        or getattr(policy, "output_dtype", None) not in (None, compute)
    ):
        return policy
    changes = {"reduce_dtype": torch.float32}
    if hasattr(policy, "output_dtype"):
        changes["output_dtype"] = None
    plugin.mixed_precision_policy = dataclasses.replace(policy, **changes)
    return plugin.mixed_precision_policy


def _parents(model):
    parents = {}
    for parent in model.modules():
        for child in parent.children():
            parents.setdefault(id(child), parent)
    return parents


def _fsdp_owner(module, parents, FSDPModule):
    """The FSDP unit whose pre-forward gathers `module`'s own parameters."""
    while module is not None:
        if isinstance(module, FSDPModule):
            return module
        module = parents.get(id(module))
    return None


def _decoder_blocks(model):
    names = getattr(model, "_no_split_modules", None) or ()
    names = set(names)
    if not names:
        return []
    # fully_shard swaps a module's class for an FSDP<Name> subclass, so match along the MRO.
    return [
        m for m in model.modules() if any(c.__name__ in names for c in type(m).__mro__)
    ]


def _all_ranks_agree(value, device):
    """True only when every rank says True, so every rank takes the same branch."""
    try:
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return value
        flag = torch.tensor([1 if value else 0], dtype = torch.int32, device = device)
        dist.all_reduce(flag, op = dist.ReduceOp.MIN)
        return bool(flag.item())
    except Exception:
        return value


def _fused_lora_wrap_problem(model, fused, parents, FSDPModule):
    """None when every fused LoRA layer runs inside the forward of a per-block FSDP unit that
    gathers all the weights it reads, else why not."""
    for module in fused:
        owner = _fsdp_owner(module, parents, FSDPModule)
        if owner is None or owner is model:
            return "FSDP2 wraps only the root module, not each decoder block"
        for sub in module.modules():
            if sub is not module and isinstance(sub, FSDPModule):
                return "FSDP2 made a submodule of a fused LoRA layer its own unit"
    return None


def _gate_fused_lora(accelerator, model, parents, FSDPModule):
    from .llama import _decline_fused_lora_for_fsdp, _fused_lora_modules

    fused = _fused_lora_modules(model)
    if not fused or os.environ.get("UNSLOTH_FORCE_FUSED_LORA", "0") == "1":
        model._unsloth_fsdp2_fused_lora = len(fused)
        return len(fused)
    problem = _fused_lora_wrap_problem(model, fused, parents, FSDPModule)
    device = getattr(accelerator, "device", None) or torch.device("cpu")
    if not _all_ranks_agree(problem is None, device):
        _decline_fused_lora_for_fsdp(model, problem or "another rank's FSDP2 wrap")
        model._unsloth_fsdp2_fused_lora = 0
        return 0
    model._unsloth_fsdp2_fused_lora = len(fused)
    _print_rank0(
        f"Unsloth: FSDP2 shards each decoder block as its own unit, so {len(fused)} fused LoRA "
        "layers stay on (their weights are gathered before the block's forward)."
    )
    return len(fused)


def _gather_units_read_directly(model, blocks, parents, FSDPModule):
    """Gather the FSDP units holding lm_head / the final norm before the model's forward."""
    if not hasattr(FSDPModule, "unshard"):
        return 0
    targets = []
    get_output = getattr(model, "get_output_embeddings", None)
    output = get_output() if callable(get_output) else None
    if isinstance(output, torch.nn.Module):
        targets.append(output)
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] in _DIRECT_READ_NORMS:
            parent = parents.get(id(module))
            if parent is not None and isinstance(
                getattr(parent, "layers", None), torch.nn.ModuleList
            ):
                targets.append(module)
    block_ids = {id(b) for b in blocks}
    units = []
    for target in targets:
        owner = _fsdp_owner(target, parents, FSDPModule)
        if owner is None or owner is model or id(owner) in block_ids:
            continue
        if all(owner is not u for u in units):
            units.append(owner)
    if not units:
        return 0
    # On the root, which the Trainer always calls through __call__ (peft's LoraModel calls the
    # CausalLM's .forward directly, skipping its hooks). Runs after FSDP's own root pre-forward.
    def _unsloth_gather_direct_reads(module, args):
        for unit in units:
            unit.unshard()

    model.register_forward_pre_hook(_unsloth_gather_direct_reads)
    return len(units)


def after_fsdp2_wrap(accelerator, model):
    """Runs once on the model accelerate just sharded."""
    FSDPModule = _fsdp_module_class()
    if FSDPModule is None or not isinstance(model, FSDPModule):
        return
    if getattr(model, "_unsloth_fsdp2_prepared", False):
        return
    model._unsloth_fsdp2_prepared = True
    parents = _parents(model)
    blocks = [b for b in _decoder_blocks(model) if isinstance(b, FSDPModule)]
    plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None)
    if (
        blocks
        and not _keep_accelerate_defaults()
        and getattr(plugin, "reshard_after_forward", True) is True
        and hasattr(blocks[-1], "set_reshard_after_forward")
    ):
        # Backward starts with the last block, so resharding it after forward only re-gathers it.
        blocks[-1].set_reshard_after_forward(False, recurse = False)
    model._unsloth_fsdp2_blocks = len(blocks)
    model._unsloth_fsdp2_direct_read_units = _gather_units_read_directly(
        model, blocks, parents, FSDPModule
    )
    _gate_fused_lora(accelerator, model, parents, FSDPModule)


def install_fsdp2_prepare_hook():
    """Wrap accelerate's `fsdp2_prepare_model` (looked up by `Accelerator._prepare_fsdp2`)."""
    try:
        import accelerate.accelerator as accelerator_module
    except Exception:
        return False
    original = getattr(accelerator_module, "fsdp2_prepare_model", None)
    if original is None:
        return False
    if getattr(original, _PATCHED, False):
        return True

    @functools.wraps(original)
    def fsdp2_prepare_model(accelerator, model, *args, **kwargs):
        fill_fsdp2_mixed_precision(accelerator)
        model = original(accelerator, model, *args, **kwargs)
        after_fsdp2_wrap(accelerator, model)
        return model

    setattr(fsdp2_prepare_model, _PATCHED, True)
    accelerator_module.fsdp2_prepare_model = fsdp2_prepare_model
    return True


def install_fsdp2_no_sync():
    """accelerate < 1.11 looks for `model.no_sync`, which FSDP2 modules do not have, so every
    gradient accumulation micro-step reduce-scattered. Map it to `set_requires_gradient_sync`."""
    try:
        from accelerate import Accelerator
    except Exception:
        return False
    original = Accelerator.no_sync
    if getattr(original, _PATCHED, False):
        return True
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError):
        return False
    if "set_requires_gradient_sync" in source:
        return True

    @contextlib.contextmanager
    def no_sync(self, model):
        if getattr(self, "is_fsdp2", False) and hasattr(model, "set_requires_gradient_sync"):
            model.set_requires_gradient_sync(False)
            try:
                yield
            finally:
                model.set_requires_gradient_sync(True)
        else:
            with original(self, model):
                yield

    no_sync.__doc__ = original.__doc__
    setattr(no_sync, _PATCHED, True)
    Accelerator.no_sync = no_sync
    return True


def fsdp2_trainer_setup(trainer):
    """Called from Trainer.__init__ once the Accelerator exists.

    Returns (keep_fused_lora, reason). keep_fused_lora is False for FSDP1 and for an FSDP2 wrap
    that will not give each decoder block its own unit; the caller declines the kernels then.
    """
    accelerator = getattr(trainer, "accelerator", None)
    plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None)
    if plugin is None or getattr(plugin, "fsdp_version", 1) != 2:
        return False, None
    install_fsdp2_prepare_hook()
    install_fsdp2_no_sync()
    if os.environ.get("UNSLOTH_FSDP2_FUSED_LORA", "1").strip().lower() in _FALSY:
        return False, "UNSLOTH_FSDP2_FUSED_LORA=0"
    if getattr(plugin, "auto_wrap_policy", None) is None:
        # accelerate's NO_WRAP: one unit for the whole model.
        return False, "FSDP2 wraps only the root module (auto_wrap_policy NO_WRAP)"
    return True, None
