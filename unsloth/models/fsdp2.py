# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""What Unsloth adds around accelerate's `fsdp2_prepare_model` (which already wraps per block,
keeps the [norm, lm_head] tail gathered, groups tied embeddings). `UNSLOTH_FSDP2_ACCELERATE_DEFAULTS=1`
skips the mixed precision fill and the last-block setting. bitsandbytes 4-bit (QLoRA) weights are
sharded too, keeping their QuantState (`fsdp2_qlora.py`).
"""

import contextlib
import dataclasses
import functools
import inspect
import os

import torch

__all__ = [
    "fsdp2_trainer_setup",
    "install_fsdp2_hooks",
    "install_fsdp2_policy_provenance",
    "install_fsdp2_prepare_hook",
    "install_fsdp2_no_sync",
    "install_fsdp2_peft_full_state_dict",
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


_AUTO_POLICY = "_unsloth_auto_mixed_precision_policy"
_POLICY_SEEN = "_unsloth_mixed_precision_policy_seen"


def install_fsdp2_policy_provenance():
    """Tag the policy accelerate builds from `mixed_precision` itself, so a user-written policy with
    the same dtypes is never mistaken for it. Must run before the Accelerator is created."""
    try:
        from accelerate.utils import FullyShardedDataParallelPlugin
    except Exception:
        return False
    original = getattr(FullyShardedDataParallelPlugin, "set_mixed_precision", None)
    if original is None:
        return False
    if getattr(original, _PATCHED, False):
        return True

    @functools.wraps(original)
    def set_mixed_precision(self, mixed_precision, *args, **kwargs):
        before = getattr(self, "mixed_precision_policy", None)
        result = original(self, mixed_precision, *args, **kwargs)
        override = kwargs.get("override", args[1] if len(args) > 1 else False)
        after = getattr(self, "mixed_precision_policy", None)
        setattr(self, _POLICY_SEEN, True)
        if before is None and not override and after is not None:
            setattr(self, _AUTO_POLICY, after)
        return result

    setattr(set_mixed_precision, _PATCHED, True)
    FullyShardedDataParallelPlugin.set_mixed_precision = set_mixed_precision
    return True


def fill_fsdp2_mixed_precision(accelerator):
    """Only rewrites accelerate's automatic policy (every dtype = compute dtype): it casts the loss
    to float16, where the 65536 loss scale overflows and the scaler skips the first steps."""
    plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None)
    policy = getattr(plugin, "mixed_precision_policy", None)
    if policy is None or _keep_accelerate_defaults():
        return policy
    automatic = getattr(plugin, _AUTO_POLICY, None) is policy
    if not automatic and getattr(plugin, _POLICY_SEEN, False):
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
    if not automatic:
        # The Accelerator was built before Unsloth was imported, so whether accelerate or the caller
        # wrote this policy is unknown. Only float16 overflows with it.
        if compute != torch.float16:
            return policy
        raise RuntimeError(
            "Unsloth: this FSDP2 Accelerator was created before `import unsloth`, so Unsloth cannot "
            "tell whether its float16 mixed precision policy (float16 gathers, reductions and "
            "outputs, which overflow at the GradScaler's initial loss scale) was chosen by "
            "accelerate or by you. Import unsloth before creating the Accelerator, or set "
            "UNSLOTH_FSDP2_ACCELERATE_DEFAULTS=1 to keep the policy as it is."
        )
    changes = {"reduce_dtype": torch.float32}
    if compute == torch.float16:
        # float16 gathers move the embedding / norm / lm_head backward into float16, where scaled
        # gradients overflow; gather the float32 masters and let autocast cast like one GPU does.
        changes["param_dtype"] = None
    if hasattr(policy, "output_dtype"):
        changes["output_dtype"] = None
    plugin.mixed_precision_policy = dataclasses.replace(policy, **changes)
    return plugin.mixed_precision_policy


def unify_trainable_dtypes(accelerator, model):
    """16-bit full finetuning runs without autocast with float32 norms, which accelerate does not
    upcast and FSDP2 refuses ("FSDP expects uniform original parameter dtype"). A policy the caller
    set is kept; only the master weights are upcast."""
    trainable = [p for p in model.parameters() if p.requires_grad and p.is_floating_point()]
    dtypes = {p.dtype for p in trainable}
    if str(getattr(accelerator, "mixed_precision", "no")) != "no":
        # accelerate upcasts these itself, except 1.13 (`param = param.to(float32)` drops the result).
        if len(dtypes) <= 1:
            return False
        for p in trainable:
            if p.dtype != torch.float32:
                p.data = p.data.to(torch.float32)
        return True
    half = torch.bfloat16 if torch.bfloat16 in dtypes else torch.float16
    if len(dtypes) <= 1 or half not in dtypes:
        return False
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy
    except ImportError:
        return False
    plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None)
    if plugin is None:
        return False
    for p in trainable:
        if p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)
    if getattr(plugin, "mixed_precision_policy", None) is None:
        plugin.mixed_precision_policy = MixedPrecisionPolicy(
            param_dtype = half, reduce_dtype = torch.float32
        )
        how = f"gathering them in {str(half).replace('torch.', '')} for compute"
    else:
        how = "gathering them as the given mixed_precision_policy says"
    _print_rank0(
        "Unsloth: FSDP2 needs one dtype per unit, and the trainable weights mix "
        f"{sorted(str(d).replace('torch.', '') for d in dtypes)}. Keeping float32 master weights and {how}."
    )
    return True


def _fsdp2_checks_frozen_dtypes():
    """torch < 2.8 asserts one original dtype over every parameter of a unit, frozen ones included
    (`FSDPParamGroup._init_mp_dtypes`); 2.8 checks only the trainable ones."""
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
        return "requires_grad" not in inspect.getsource(FSDPParamGroup._init_mp_dtypes)
    except Exception:
        pass
    try:
        major, minor = (int(x) for x in torch.__version__.split(".")[:2])
        return (major, minor) < (2, 8)
    except Exception:
        return False


def unify_frozen_dtypes(
    accelerator,
    model,
    frozen_checked = None,
):
    """LoRA on torch < 2.8: float32 adapters next to frozen 16-bit weights in one unit. Upcast the
    frozen weights too (what accelerate <= 1.12 did with `model.to(float32)`); the gather policy
    casts them back down for compute."""
    if frozen_checked is None:
        frozen_checked = _fsdp2_checks_frozen_dtypes()
    if not frozen_checked:
        return False
    dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
    if len(dtypes) <= 1:
        return False
    plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None)
    if plugin is None:
        return False
    if (
        getattr(plugin, "mixed_precision_policy", None) is None
        and str(getattr(accelerator, "mixed_precision", "no")) == "no"
    ):
        raise RuntimeError(
            f"Unsloth: FSDP2 on torch {torch.__version__} needs every parameter of a unit to share "
            f"one dtype, but this model mixes {sorted(str(d) for d in dtypes)} and no mixed "
            "precision is set to compute in 16-bit from float32 weights. Enable bf16 / fp16 mixed "
            "precision, or use torch >= 2.8."
        )
    n = 0
    for p in model.parameters():
        # Quantized parameter subclasses keep their storage dtype.
        if type(p) is torch.nn.Parameter and p.is_floating_point() and p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)
            n += 1
    if n:
        _print_rank0(
            f"Unsloth: FSDP2 on torch {torch.__version__} needs one dtype per unit, frozen weights "
            f"included, so {n} 16-bit parameters are kept in float32 and gathered in the compute dtype."
        )
    return n > 0


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
    return [m for m in model.modules() if any(c.__name__ in names for c in type(m).__mro__)]


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


def _in_decoder_block(module, block_ids, parents):
    while module is not None:
        parent = parents.get(id(module))
        if id(module) in block_ids or isinstance(parent, torch.nn.ModuleList):
            return True
        module = parent
    return False


def _fused_lora_wrap_problem(
    model,
    fused,
    parents,
    FSDPModule,
    blocks = (),
):
    """None when every fused layer's weights are gathered by a per-block unit, else why not."""
    block_ids = {id(b) for b in blocks}
    for module in fused:
        owner = _fsdp_owner(module, parents, FSDPModule)
        if owner is None or owner is model:
            return "FSDP2 wraps only the root module, not each decoder block"
        if not _in_decoder_block(owner, block_ids, parents):
            # Blocks are always entered through __call__; a unit above them need not be (peft's
            # LoraModel calls the CausalLM's .forward), so its pre-forward gather can be skipped.
            return "FSDP2 wraps a module above the decoder blocks, not each block"
        for sub in module.modules():
            if sub is not module and isinstance(sub, FSDPModule):
                return "FSDP2 made a submodule of a fused LoRA layer its own unit"
    return None


def _gate_fused_lora(
    accelerator,
    model,
    parents,
    FSDPModule,
    blocks = None,
):
    from .llama import _decline_fused_lora_for_fsdp, _fused_lora_modules

    fused = _fused_lora_modules(model)
    if not fused or os.environ.get("UNSLOTH_FORCE_FUSED_LORA", "0") == "1":
        model._unsloth_fsdp2_fused_lora = len(fused)
        return len(fused)
    if os.environ.get("UNSLOTH_FSDP2_FUSED_LORA", "1").strip().lower() in _FALSY:
        # A raw loop patches the model before any FSDP plugin exists, so load time could not see it.
        problem = "UNSLOTH_FSDP2_FUSED_LORA=0"
    else:
        if blocks is None:
            blocks = _decoder_blocks(model)
        problem = _fused_lora_wrap_problem(model, fused, parents, FSDPModule, blocks)
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
    """Unsloth reads lm_head / final norm weights directly (fused CE, fast RMSNorm), so FSDP's
    pre-forward never gathers their unit (accelerate >= 1.14 splits it off)."""
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

    # On the root: peft's LoraModel calls the CausalLM's .forward directly, skipping its hooks.
    def _unsloth_gather_direct_reads(module, args):
        for unit in units:
            unit.unshard()

    model.register_forward_pre_hook(_unsloth_gather_direct_reads)
    return len(units)


def after_fsdp2_wrap(accelerator, model):
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
    _gate_fused_lora(accelerator, model, parents, FSDPModule, blocks)
    from .fsdp2_save import reduce_unsharded_trainable_grads

    replicated = reduce_unsharded_trainable_grads(model)
    if replicated:
        _print_rank0(
            f"Unsloth: FSDP2 leaves {len(replicated)} trainable parameters unsharded (e.g. "
            f"{replicated[0]}); their gradients are averaged over ranks like DDP."
        )


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
        from .fsdp2_qlora import fsdp2_qlora_wrap

        # The 4-bit fallback to uint8 storage first: float-packed weights would count as frozen
        # 16-bit weights in the dtype unification.
        with fsdp2_qlora_wrap(model, original) as qlora:
            fill_fsdp2_mixed_precision(accelerator)
            unify_trainable_dtypes(accelerator, model)
            unify_frozen_dtypes(accelerator, model)
            model = original(accelerator, model, *args, **kwargs)
        if qlora is not None:
            note = "" if qlora["patched"] else f" FSDPParam was not patched: {qlora['reason']}."
            if qlora.get("reverted"):
                note += f" {qlora['reverted']} float-packed weights kept as uint8, which this wrap preserves."
            _print_rank0(
                f"Unsloth: FSDP2 QLoRA: {qlora['params4bit']} 4-bit weights packed as "
                f"{'/'.join(qlora['storage'])} ({qlora['converted']} repacked from uint8).{note}"
            )
        after_fsdp2_wrap(accelerator, model)
        return model

    setattr(fsdp2_prepare_model, _PATCHED, True)
    accelerator_module.fsdp2_prepare_model = fsdp2_prepare_model
    return True


def install_fsdp2_no_sync():
    """accelerate < 1.11 looks for `model.no_sync`, which FSDP2 modules lack."""
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


def install_fsdp2_peft_full_state_dict():
    """accelerate < 1.15 builds an FSDP2 PEFT model's FULL_STATE_DICT checkpoint (what Trainer writes for
    resuming) from `get_peft_model_state_dict`, i.e. each rank's DTensor shard, and rank 0 saves only
    its own. Backport of the 1.15 fix: gather the trainable weights with `get_model_state_dict`."""
    try:
        import accelerate.utils.fsdp_utils as fsdp_utils
    except Exception:
        return False
    original_get = getattr(fsdp_utils, "_get_model_state_dict", None)
    original_set = getattr(fsdp_utils, "_set_model_state_dict", None)
    is_peft_model = getattr(fsdp_utils, "is_peft_model", None)
    if original_get is None or original_set is None or is_peft_model is None:
        return False
    if getattr(original_get, _PATCHED, False):
        return True
    try:
        source = inspect.getsource(original_get)
        parameters = inspect.signature(original_get).parameters
    except (OSError, TypeError, ValueError):
        return False
    if "full_state_dict" in source or "sd_options" not in parameters:
        return True

    def _full_peft(model, adapter_only, sd_options):
        return (
            adapter_only
            and sd_options is not None
            and getattr(sd_options, "full_state_dict", False)
            and is_peft_model(model)
        )

    @functools.wraps(original_get)
    def _get_model_state_dict(
        model,
        adapter_only = False,
        sd_options = None,
    ):
        if _full_peft(model, adapter_only, sd_options):
            from torch.distributed.checkpoint.state_dict import get_model_state_dict
            options = dataclasses.replace(sd_options, ignore_frozen_params = True)
            return get_model_state_dict(model, options = options)
        return original_get(model, adapter_only = adapter_only, sd_options = sd_options)

    @functools.wraps(original_set)
    def _set_model_state_dict(
        model,
        state_dict,
        adapter_only = False,
        sd_options = None,
    ):
        if _full_peft(model, adapter_only, sd_options):
            from torch.distributed.checkpoint.state_dict import set_model_state_dict
            options = dataclasses.replace(sd_options, strict = False)
            return set_model_state_dict(model, state_dict, options = options)
        return original_set(model, state_dict, adapter_only = adapter_only, sd_options = sd_options)

    setattr(_get_model_state_dict, _PATCHED, True)
    setattr(_set_model_state_dict, _PATCHED, True)
    fsdp_utils._get_model_state_dict = _get_model_state_dict
    fsdp_utils._set_model_state_dict = _set_model_state_dict
    return True


def install_fsdp2_hooks():
    """Everything FSDP2 needs before an Accelerator exists; also for raw `Accelerator.prepare()` loops."""
    install_fsdp2_policy_provenance()
    install_fsdp2_prepare_hook()
    install_fsdp2_no_sync()
    install_fsdp2_peft_full_state_dict()
    from .fsdp2_save import install_dcp_bnb_state_dict

    install_dcp_bnb_state_dict()


def fsdp2_trainer_setup(trainer):
    """Returns (keep_fused_lora, reason); False for FSDP1 and for a root-only FSDP2 wrap."""
    accelerator = getattr(trainer, "accelerator", None)
    plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None)
    if plugin is None or getattr(plugin, "fsdp_version", 1) != 2:
        return False, None
    install_fsdp2_prepare_hook()
    install_fsdp2_no_sync()
    install_fsdp2_peft_full_state_dict()
    from .fsdp2_save import install_dcp_bnb_state_dict

    install_dcp_bnb_state_dict()
    if os.environ.get("UNSLOTH_FSDP2_FUSED_LORA", "1").strip().lower() in _FALSY:
        return False, "UNSLOTH_FSDP2_FUSED_LORA=0"
    if getattr(plugin, "auto_wrap_policy", None) is None:
        return False, "FSDP2 wraps only the root module (auto_wrap_policy NO_WRAP)"
    return True, None
