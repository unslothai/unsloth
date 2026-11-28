# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""bitsandbytes optimizer base classes for Unsloth's bitsandbytes stand-in.

Optimizer8bit / Optimizer2State follow bitsandbytes' own classes: the same constructor arguments,
param-group config, GlobalOptimManager overrides, state layout and names (step, state1, state2,
qmap1, qmap2, absmax1, absmax2), the same state_dict (8-bit tensors wrapped for FSDP), so an
optimizer checkpoint moves between this and bitsandbytes in both directions. Adam / AdamW steps run
Unsloth's Triton kernels (unsloth/kernels/adam8bit_blockwise.py), bit-identical to bitsandbytes'
CUDA kernels on NVIDIA GPUs, or the plain torch version of the same update on CPU tensors and
where Triton is unavailable. 8-bit state is blockwise (256 values per absmax), about 2 bytes per
parameter; weights under min_8bit_size elements (and GlobalOptimManager optim_bits=32 overrides)
keep 32-bit state, as in bitsandbytes.

Paged optimizers (is_paged=True) keep their state in ordinary device memory: paging is part of
bitsandbytes' native library. Optimizers without a counterpart here (SGD / RMSprop / Adagrad)
build the plain torch optimizer after a one time notice; Lion, AdEMAMix, LAMB and LARS refuse."""

import os
from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain

import torch

from .. import functional as F
from ._adam_torch import adam_8bit_blockwise_torch, adam_32bit_torch

__all__ = [
    "GlobalOptimManager",
    "MockArgs",
    "Optimizer1State",
    "Optimizer2State",
    "Optimizer8bit",
    "torch_optimizer_fallback",
]


class MockArgs:
    def __init__(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])


class GlobalOptimManager:
    """Per-parameter config overrides (transformers' Trainer pins embeddings to 32-bit state)."""

    _instance = None

    def __init__(self):
        raise RuntimeError("Call get_instance() instead")

    def initialize(self):
        self.pid2config = {}
        self.index2config = {}
        self.optimizer = None
        self.uses_config_override = False
        self.module_weight_config_triple = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def register_parameters(self, params):
        param_groups = list(params)
        if not param_groups:
            return
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]
        for group_index, group in enumerate(param_groups):
            for p_index, p in enumerate(group["params"]):
                if id(p) in self.pid2config:
                    self.index2config[(group_index, p_index)] = self.pid2config[id(p)]

    def override_config(
        self,
        parameters,
        key = None,
        value = None,
        key_value_dict = None,
    ):
        self.uses_config_override = True
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        if key is not None and value is not None:
            assert key_value_dict is None
            key_value_dict = {key: value}
        if key_value_dict is not None:
            for p in parameters:
                if id(p) in self.pid2config:
                    self.pid2config[id(p)].update(key_value_dict)
                else:
                    self.pid2config[id(p)] = key_value_dict

    def register_module_override(self, module, param_name, config):
        self.module_weight_config_triple.append((module, param_name, config))


# UNSLOTH_BNB_SHIM_TRITON=0 steps with the torch path everywhere; read once, not inside traced code.
_TRITON_WANTED = os.environ.get("UNSLOTH_BNB_SHIM_TRITON", "1") != "0"
_KERNELS = {"module": None, "ok": None}


def _triton_kernels(device):
    """unsloth/kernels/adam8bit_blockwise.py when it can run on `device`, else None."""
    if not _TRITON_WANTED or device.type != "cuda" or _KERNELS["ok"] is False:
        return None
    if _KERNELS["ok"] is None:
        if torch.compiler.is_compiling():
            return None
        try:
            from unsloth.kernels import adam8bit_blockwise
            _KERNELS["module"] = adam8bit_blockwise
            _KERNELS["ok"] = True
        except Exception as error:
            _KERNELS["ok"] = False
            print(
                f"Unsloth: Triton 8-bit Adam kernels are unavailable ({type(error).__name__}: "
                f"{error}); the bitsandbytes stand-in steps Adam with torch ops."
            )
            return None
    return _KERNELS["module"]


_BF16_OK = {}


def _bf16_triton_ok(device):
    """Triton refuses bfloat16 below sm80 (T4): such weights take the torch path there."""
    if torch.version.hip is not None:
        return True
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _BF16_OK:
        _BF16_OK[index] = torch.cuda.get_device_capability(index)[0] >= 8
    return _BF16_OK[index]


def _triton_failed(error):
    """A compile / launch failure (not OOM, not a tracing error) moves every later step to torch."""
    if isinstance(error, torch.cuda.OutOfMemoryError) or torch.compiler.is_compiling():
        return False
    if type(error).__module__.startswith("torch._dynamo"):
        return False
    _KERNELS["ok"] = False
    print(
        f"Unsloth: the Triton 8-bit Adam step failed ({type(error).__name__}: {error}); the "
        "bitsandbytes stand-in steps Adam with torch ops from now on."
    )
    return True


def _torch_moments_to_32bit(entry):
    """A checkpoint from torch Adam / AdamW (what the stand-in ran before it kept 8-bit state, or
    any torch run): its moments become this optimizer's 32-bit state, same update rule from there."""
    if not isinstance(entry, dict) or "state1" in entry:
        return
    m, v = entry.get("exp_avg"), entry.get("exp_avg_sq")
    if not (torch.is_tensor(m) and torch.is_tensor(v)):
        return
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        entry.pop(key, None)
    entry["state1"] = m.float().contiguous()
    entry["state2"] = v.float().contiguous()
    step = entry.get("step", 0)
    entry["step"] = int(step.item() if torch.is_tensor(step) else step)


def _flat_state(t, p):
    """State shaped like p in memory order: bitsandbytes' kernels index state buffers flat, so a
    non-contiguous buffer (zeros_like of a transposed weight in a bitsandbytes checkpoint) is read
    in storage order, never permuted."""
    if t.is_contiguous():
        return t
    n = t.numel()
    storage = t.untyped_storage()
    if t.storage_offset() == 0 and storage.nbytes() == n * t.element_size():
        return torch.empty(0, dtype = t.dtype, device = t.device).set_(storage, 0, tuple(p.shape))
    return t.contiguous()


def _multi_tensor_wanted():
    # UNSLOTH_BNB_SHIM_MULTI_TENSOR=0 steps every weight with its own launch (update_step).
    return os.environ.get("UNSLOTH_BNB_SHIM_MULTI_TENSOR", "1") != "0"


_PLANS = "_unsloth_multi_tensor_plans"
_STEPS = "_unsloth_multi_tensor_steps"


def _multi_tensor_step(optimizer):
    """Every weight of the step in one launch per (weight dtype, state kind): bitsandbytes' update,
    bit for bit, without a launch per tensor. Returns False (nothing changed) when the optimizer or a
    weight is not one it covers; then each weight takes update_step as before. Covered: plain
    contiguous CUDA weights and gradients of one device, float32 / float16 / bfloat16, Python int
    steps (not capturable), at most 4 param groups, no instance-level or subclass override of
    update_step / init_state / get_config (bfloat16 stochastic rounding, FSDP2 shards keep theirs)."""
    if not _multi_tensor_wanted() or getattr(optimizer, "capturable", False):
        return False
    cls = type(optimizer)
    if (
        not isinstance(optimizer, Optimizer2State)
        or optimizer.optimizer_name != "adam"
        or any(k in vars(optimizer) for k in ("update_step", "init_state", "get_config"))
        or cls.update_step is not Optimizer2State.update_step
        or cls.init_state is not Optimizer2State.init_state
        or cls.get_config is not Optimizer8bit.get_config
        or torch.compiler.is_compiling()
    ):
        return False
    groups = optimizer.param_groups
    if len(groups) > 4:
        return False
    active = []
    device = None
    for gindex, group in enumerate(groups):
        for pindex, p in enumerate(group["params"]):
            g = p.grad
            if g is None:
                continue
            if (
                type(p) not in (torch.nn.Parameter, torch.Tensor)
                or type(g) is not torch.Tensor
                or p.device.type != "cuda"
                or g.device != p.device
                or (device is not None and p.device != device)
                or p.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                or g.dtype != p.dtype
                or g.layout != torch.strided
                or not p.is_contiguous()
                or not g.is_contiguous()
            ):
                return False
            device = p.device
            active.append((gindex, pindex, group, p))
    if not active:
        return False
    if any(p.dtype is torch.bfloat16 for _, _, _, p in active) and not _bf16_triton_ok(device):
        return False
    kernels = _triton_kernels(device)
    if kernels is None:
        return False
    # Per-group scalars; a per-parameter override of any of them keeps the per-tensor route.
    scalars = {}
    for gindex, pindex, group, p in active:
        config = optimizer.get_config(gindex, pindex, group)
        if len(config["betas"]) != 2 or config["max_unorm"] > 0.0 or config["skip_zeros"]:
            return False
        lr = config["lr"]
        value = (
            float(lr) if torch.is_tensor(lr) else lr,
            *config["betas"],
            config["eps"],
            config["weight_decay"],
        )
        if scalars.setdefault(gindex, value) != value:
            return False
    for gindex, pindex, group, p in active:
        if len(optimizer.state[p]) == 0:
            optimizer.init_state(group, p, gindex, pindex)
    parts = {}
    for gindex, pindex, group, p in active:
        st = optimizer.state[p]
        if torch.is_tensor(st.get("step")) or isinstance(st.get("step"), bool):
            return False
        eight = st["state1"].dtype == torch.uint8
        keys = (
            ("state1", "state2", "absmax1", "absmax2", "qmap1", "qmap2")
            if eight
            else ("state1", "state2")
        )
        for k in keys:
            v = st.get(k)
            if not torch.is_tensor(v) or v.device != device or not v.is_contiguous():
                return False
        parts.setdefault((p.dtype, eight), []).append((gindex, p, st))
    plans = optimizer.__dict__.setdefault(_PLANS, {})
    if any(
        plan.members_ids != _active_ids(parts.get(key, ())) for key, plan in plans.items()
    ) or set(parts) - set(plans):
        # (Re)building: one launch updates its weights concurrently, so no weight may appear twice
        # or share memory with another (the per-tensor route steps them one after another).
        if _overlapping([p for _, _, _, p in active]):
            return False
    order = sorted(scalars)
    group_scalars = [scalars[g] for g in order]
    slot = {g: i for i, g in enumerate(order)}
    launches = []
    for key, members in parts.items():
        plan = plans.get(key)
        if plan is None or not _plan_matches(plan, members, slot):
            if key[1] and not _same_qmaps(members):
                return False
            try:
                plan = kernels.MultiTensorPlan(
                    [p for _, p, _ in members],
                    [st for _, _, st in members],
                    [slot[g] for g, _, _ in members],
                    key[1],
                )
            except torch.cuda.OutOfMemoryError:
                plans.clear()
                return False
            plan.members = [
                [p, st, slot[g], _state_ids(st), int(st["step"]), p.data_ptr()]
                for g, p, st in members
            ]
            plan.members_ids = _active_ids(members)
            plans[key] = plan
        launches.append((plan, members))
    for plan, members in launches:
        try:
            plan.set_grads([p.grad for _, p, _ in members])
            plan.launch(group_scalars)
        except Exception as error:
            # A failed launch leaves this plan's device step counts unknown: rebuild next time.
            plans.clear()
            if not _triton_failed(error):
                raise
            return False if plan is launches[0][0] else _finish_rest(optimizer, launches, plan)
        for member in plan.members:
            member[1]["step"] += 1
            member[4] += 1
    optimizer.__dict__[_STEPS] = optimizer.__dict__.get(_STEPS, 0) + 1
    return True


def _active_ids(members):
    return tuple(id(p) for _, p, _ in members)


def _overlapping(params):
    """Any weight listed twice, or two weights whose memory overlaps."""
    if len({id(p) for p in params}) != len(params):
        return True
    spans = sorted(
        (p.data_ptr(), p.data_ptr() + p.numel() * p.element_size()) for p in params if p.numel()
    )
    return any(spans[i][0] < spans[i - 1][1] for i in range(1, len(spans)))


def _state_ids(st):
    return tuple(
        id(st.get(k)) for k in ("state1", "state2", "absmax1", "absmax2", "qmap1", "qmap2")
    )


def _plan_matches(plan, members, slot):
    if len(plan.members) != len(members):
        return False
    for (p0, st0, g0, ids0, step0, ptr0), (g, p, st) in zip(plan.members, members):
        if (
            p0 is not p
            or st0 is not st
            or g0 != slot[g]
            or ids0 != _state_ids(st)
            or step0 != st["step"]
            or ptr0 != p.data_ptr()
        ):
            return False
    return True


def _same_qmaps(members):
    q1, q2 = members[0][2]["qmap1"], members[0][2]["qmap2"]
    for _, _, st in members[1:]:
        if st["qmap1"] is not q1 and not torch.equal(st["qmap1"], q1):
            return False
        if st["qmap2"] is not q2 and not torch.equal(st["qmap2"], q2):
            return False
    return True


def _finish_rest(optimizer, launches, failed):
    """A later launch failed after earlier ones stepped their weights: step the rest per tensor."""
    done = set()
    for plan, members in launches:
        if plan is failed:
            break
        done.update(id(p) for _, p, _ in members)
    for gindex, group in enumerate(optimizer.param_groups):
        for pindex, p in enumerate(group["params"]):
            if p.grad is not None and id(p) not in done:
                optimizer.update_step(group, p, gindex, pindex)
    return True


class Optimizer8bit(torch.optim.Optimizer):
    _FSDP_WRAPPED_QUANT_STATE_KEY = "__bnb_optimizer_quant_state__"

    def __init__(
        self,
        params,
        defaults,
        optim_bits = 32,
        is_paged = False,
    ):
        super().__init__(params, defaults)
        self.initialized = False
        self.name2qmap = {}
        # Paged state lives in ordinary device memory here (bitsandbytes pages via its native library).
        self.is_paged = is_paged
        self.page_mng = None
        self.mng = GlobalOptimManager.get_instance()
        self.non_castable_tensor_keys = {
            "qmap1",
            "qmap2",
            "max1",
            "max2",
            "new_max1",
            "new_max2",
            "state1",
            "state2",
            "gnorm_vec",
            "absmax1",
            "absmax2",
            "unorm_vec",
        }
        if optim_bits == 8:
            self.fill_qmap()

    def fill_qmap(self):
        self.name2qmap["dynamic"] = F.create_dynamic_map(signed = True)
        self.name2qmap["udynamic"] = F.create_dynamic_map(signed = False)

    def state_dict(self):
        """bitsandbytes' layout: the 8-bit tensors of each parameter wrapped in one nested dict
        (hidden from FSDP's gather), the step a Python int."""
        state_dict = super().state_dict()
        state_dict["state"] = {
            k: {kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in state_dict["state"].items()
        }
        for param_state in state_dict["state"].values():
            if isinstance(param_state, dict):
                step = param_state.get("step")
                if torch.is_tensor(step):  # capturable: bitsandbytes counts steps in an int
                    param_state["step"] = int(step.item())
                quant_state = {}
                for key in [k for k in param_state if k in self.non_castable_tensor_keys]:
                    quant_state[key] = param_state.pop(key)
                if quant_state:
                    param_state[self._FSDP_WRAPPED_QUANT_STATE_KEY] = quant_state
        return state_dict

    def __setstate__(self, state):
        super().__setstate__(state)

    def load_state_dict(
        self,
        state_dict,
        move_to_device = True,
    ):
        state_dict = deepcopy(state_dict)
        for param_state in state_dict["state"].values():
            if isinstance(param_state, dict) and self._FSDP_WRAPPED_QUANT_STATE_KEY in param_state:
                param_state.update(param_state.pop(self._FSDP_WRAPPED_QUANT_STATE_KEY))
        groups = self.param_groups
        saved_groups = state_dict["param_groups"]
        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group that doesn't match the size of "
                "optimizer's group",
            )
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        }

        def cast(param, value):
            if isinstance(value, torch.Tensor):
                if param.is_floating_point() and value.dtype != torch.uint8:
                    value = value.to(param.dtype)
                return value
            elif isinstance(value, dict):
                for k, v in value.items():
                    if k in self.non_castable_tensor_keys:
                        if move_to_device:
                            value[k] = v.to(param.device)
                    else:
                        value[k] = cast(param, v)
                return value
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            else:
                return value

        state = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                _torch_moments_to_32bit(v)
                state[param] = cast(param, v)
                self._load_step(param, state[param])
            else:
                state[k] = v

        def update_group(group, new_group):
            new_group["params"] = group["params"]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": state, "param_groups": param_groups})

    def _load_step(self, param, entry):
        # bitsandbytes counts in a Python int; a float tensor step (torch-style) is read as one.
        step = entry.get("step") if isinstance(entry, dict) else None
        if torch.is_tensor(step) and not getattr(self, "capturable", False):
            entry["step"] = int(step.item())
        elif getattr(self, "capturable", False) and step is not None:
            entry["step"] = torch.tensor(
                float(step.item() if torch.is_tensor(step) else step),
                dtype = torch.float32,
                device = param.device,
            )

    def to_gpu(self):
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.device.type == "cpu":
                    continue
                if p in self.state:
                    values = self.state[p]
                    for k, v in values.items():
                        if isinstance(v, torch.Tensor):
                            self.state[p][k] = v.to(p.device)

    def check_overrides(self):
        for module, attr, config in self.mng.module_weight_config_triple:
            pmodule = getattr(module, attr)
            assert pmodule is not None
            assert isinstance(pmodule, torch.Tensor)
            found = False
            for gindex, group in enumerate(self.param_groups):
                if found:
                    break
                for pindex, p in enumerate(group["params"]):
                    if found:
                        break
                    if id(p) == id(pmodule):
                        self.mng.pid2config[id(p)] = config
                        self.mng.index2config[(gindex, pindex)] = self.mng.pid2config[id(p)]
                        found = True

    # Eager, like bitsandbytes' ctypes step: a compiled caller graph-breaks here once instead of
    # recompiling on every Python int step count. Traced code reaches the kernels as opaque ops.
    @torch._disable_dynamo
    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self.initialized:
            self.check_overrides()
            self.to_gpu()
            self.initialized = True
        if _multi_tensor_step(self):
            return loss
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    self.init_state(group, p, gindex, pindex)
                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)
        # No per-parameter device sync: the kernels run on the current stream (nothing is paged).
        return loss

    def get_config(self, gindex, pindex, group):
        config = {}
        config["betas"] = group["betas"]
        config["eps"] = group["eps"]
        config["weight_decay"] = group["weight_decay"]
        config["lr"] = group["lr"]
        config["alpha"] = group.get("alpha", 0.0)
        config["t_alpha"] = group.get("t_alpha", None)
        config["t_beta3"] = group.get("t_beta3", None)
        config["optim_bits"] = self.args.optim_bits
        config["min_8bit_size"] = self.args.min_8bit_size
        config["max_unorm"] = self.args.max_unorm
        config["skip_zeros"] = self.args.skip_zeros
        if (gindex, pindex) in self.mng.index2config:
            config.update(self.mng.index2config[(gindex, pindex)])
        p = self.param_groups[gindex]["params"][pindex]
        if id(p) in self.mng.pid2config:
            config.update(self.mng.pid2config[id(p)])
        return config

    def init_state(self, group, p, gindex, pindex):
        raise NotImplementedError("init_state method needs to be overridden")

    def update_step(self, group, p, gindex, pindex):
        raise NotImplementedError("The update_step method needs to be overridden")

    def get_state_buffer(
        self,
        p,
        dtype = torch.float32,
    ):
        # Paged or not: ordinary memory on the parameter's device, contiguous whatever p's strides
        # (the kernels index state flat, as bitsandbytes' do).
        return torch.zeros(p.shape, dtype = dtype, device = p.device)

    def prefetch_state(self, p):
        return None


# Constructor arguments of older bitsandbytes releases: accepted at their defaults only.
_LEGACY_DEFAULTS = {"percentile_clipping": 100, "block_wise": True}


def _check_legacy(name, kwargs):
    for key, value in list(kwargs.items()):
        if key not in _LEGACY_DEFAULTS:
            raise TypeError(f"{name}.__init__() got an unexpected keyword argument '{key}'")
        if value != _LEGACY_DEFAULTS[key]:
            raise ValueError(
                f"Unsloth: `{name}({key}={value!r})` needs the bitsandbytes package; the stand-in "
                f"supports {key}={_LEGACY_DEFAULTS[key]!r}."
            )


class Optimizer2State(Optimizer8bit):
    def __init__(
        self,
        optimizer_name,
        params,
        lr = 1e-3,
        betas = (0.9, 0.999),
        eps = 1e-8,
        weight_decay = 0.0,
        optim_bits = 32,
        args = None,
        min_8bit_size = 4096,
        max_unorm = 0.0,
        skip_zeros = False,
        is_paged = False,
        alpha = 0.0,
        t_alpha = None,
        t_beta3 = None,
        capturable = False,
        **legacy,
    ):
        """bitsandbytes' Optimizer2State; only optimizer_name="adam" (Adam / AdamW) steps here.
        capturable=True (Unsloth addition) keeps the step count on the device and reads it, and a
        tensor learning rate, inside the kernel, so the step can be captured in a CUDA graph."""
        _check_legacy(type(self).__name__, legacy)
        if optimizer_name != "adam":
            raise RuntimeError(
                f"Unsloth: the bitsandbytes stand-in steps Adam / AdamW only, not "
                f"`{optimizer_name}`. Install bitsandbytes for it."
            )
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if isinstance(betas, str):
            betas = betas.replace("(", "").replace(")", "").strip().split(",")
            betas = [float(b) for b in betas]
        for i in range(len(betas)):
            if not 0.0 <= betas[i] < 1.0:
                raise ValueError(f"Invalid beta parameter at index {i}: {betas[i]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if max_unorm > 0.0 or skip_zeros:
            raise ValueError(
                "Unsloth: max_unorm and skip_zeros need the bitsandbytes package; the stand-in "
                "supports max_unorm=0.0, skip_zeros=False."
            )
        defaults = dict(
            lr = lr,
            betas = betas,
            eps = eps,
            weight_decay = weight_decay,
            alpha = alpha,
            t_alpha = t_alpha,
            t_beta3 = t_beta3,
        )
        super().__init__(params, defaults, optim_bits, is_paged)
        if args is None:
            args = {}
            args["optim_bits"] = optim_bits
            args["min_8bit_size"] = min_8bit_size
            args["max_unorm"] = max_unorm
            args["skip_zeros"] = skip_zeros
            args["percentile_clipping"] = 100
            args["block_wise"] = True
            self.args = MockArgs(args)
        else:
            self.args = args
        self.optimizer_name = optimizer_name
        self.capturable = bool(capturable)

    def _device_lr(self, gindex, lr, device):
        # capturable: the kernel reads the rate from device memory. A float fills a per-group
        # buffer (inside a captured graph that value is fixed; pass a tensor lr to schedule it).
        if torch.is_tensor(lr):
            return lr if lr.device == device else lr.to(device)
        bufs = self.__dict__.setdefault("_unsloth_lr_buffers", {})
        buf = bufs.get((gindex, device))
        if buf is None:
            buf = bufs[(gindex, device)] = torch.empty((), dtype = torch.float32, device = device)
        return buf.fill_(lr)

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        config = self.get_config(gindex, pindex, group)
        if config["optim_bits"] == 32:
            dtype = torch.float32
        elif config["optim_bits"] == 8:
            dtype = torch.uint8
        else:
            raise NotImplementedError(
                f"Amount of optimizer bits not supported: {config['optim_bits']}"
            )
        if p.numel() < config["min_8bit_size"]:
            dtype = torch.float32
        state = self.state[p]
        state["step"] = (
            torch.zeros((), dtype = torch.float32, device = p.device) if self.capturable else 0
        )
        if dtype == torch.float32:
            state["state1"] = self.get_state_buffer(p, dtype = torch.float32)
            state["state2"] = self.get_state_buffer(p, dtype = torch.float32)
        else:
            if "dynamic" not in self.name2qmap:
                self.fill_qmap()
            self.name2qmap["dynamic"] = self.name2qmap["dynamic"].to(p.device)
            self.name2qmap["udynamic"] = self.name2qmap["udynamic"].to(p.device)
            state["state1"] = self.get_state_buffer(p, dtype = torch.uint8)
            state["qmap1"] = self.name2qmap["dynamic"]
            state["state2"] = self.get_state_buffer(p, dtype = torch.uint8)
            state["qmap2"] = self.name2qmap["udynamic"]
            blocksize = 256
            n = p.numel()
            blocks = (n // blocksize) + bool(n % blocksize)
            state["absmax1"] = torch.zeros((blocks,), dtype = torch.float32, device = p.device)
            state["absmax2"] = torch.zeros((blocks,), dtype = torch.float32, device = p.device)

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        p.data = p.data.contiguous()
        p.grad = p.grad.contiguous()
        state = self.state[p]
        grad = p.grad
        for key in ("state1", "state2"):
            if not state[key].is_contiguous():
                state[key] = _flat_state(state[key], p)
        config = self.get_config(gindex, pindex, group)
        if p.dtype not in (torch.float32, torch.float16, torch.bfloat16) or grad.dtype != p.dtype:
            raise ValueError(
                f"Unsloth: bitsandbytes Adam needs float32 / float16 / bfloat16 weights with "
                f"gradients of the same dtype, got weight {p.dtype} and gradient {grad.dtype}"
            )
        if torch.is_tensor(state["step"]):
            state["step"].add_(1)
        else:
            state["step"] += 1
        step = state["step"]
        beta1, beta2 = config["betas"][0], config["betas"][1]
        eps, lr, weight_decay = config["eps"], config["lr"], config["weight_decay"]
        if torch.is_tensor(step):
            lr = self._device_lr(gindex, lr, p.device)
        elif torch.is_tensor(lr):
            lr = float(lr)  # bitsandbytes reads a tensor learning rate as a float too
        eight = state["state1"].dtype == torch.uint8
        kernels = _triton_kernels(p.device)
        if kernels is not None and p.dtype is torch.bfloat16 and not _bf16_triton_ok(p.device):
            kernels = None
        if kernels is not None:
            try:
                if eight:
                    kernels.compiled_safe_8bit_step(
                        p,
                        grad,
                        state["state1"],
                        state["state2"],
                        state["qmap1"],
                        state["qmap2"],
                        state["absmax1"],
                        state["absmax2"],
                        step,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                    )
                else:
                    kernels.compiled_safe_32bit_step(
                        p,
                        grad,
                        state["state1"],
                        state["state2"],
                        step,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                    )
                return
            except Exception as error:
                if not _triton_failed(error):
                    raise
        if eight:
            adam_8bit_blockwise_torch(
                p,
                grad,
                state["state1"],
                state["state2"],
                state["qmap1"],
                state["qmap2"],
                state["absmax1"],
                state["absmax2"],
                step,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            )
        else:
            adam_32bit_torch(
                p, grad, state["state1"], state["state2"], step, lr, beta1, beta2, eps, weight_decay
            )


class Optimizer1State(Optimizer8bit):
    """isinstance target: one-state 8-bit optimizers (Lion, SGD momentum, ...) need bitsandbytes."""

    def __init__(
        self,
        optimizer_name = None,
        *args,
        **kwargs,
    ):
        raise RuntimeError(
            f"Unsloth: the bitsandbytes stand-in has no one-state 8-bit optimizer "
            f"(`{optimizer_name}`). Use bitsandbytes.optim.AdamW8bit / optim='adamw_8bit', or "
            "install bitsandbytes."
        )


_NOTICED = set()
# bitsandbytes-only constructor arguments, dropped for the torch optimizer.
_BNB_ONLY = (
    "optim_bits",
    "args",
    "min_8bit_size",
    "percentile_clipping",
    "block_wise",
    "is_paged",
    "max_unorm",
    "skip_zeros",
)
# Keys only bitsandbytes optimizer state has: such a checkpoint cannot resume a torch optimizer.
_BNB_STATE_KEYS = ("state1", "state2", "qmap1", "qmap2", "absmax1", "absmax2")


def _materialize(params):
    """List of parameters or of param-group dicts, every generator consumed exactly once."""
    out = []
    for item in params:
        if isinstance(item, dict):
            group = dict(item)
            group_params = group.get("params", ())
            if isinstance(group_params, torch.Tensor):
                group_params = [group_params]
            group["params"] = list(group_params)
            out.append(group)
        else:
            out.append(item)
    return out


def _parameters(params):
    for item in params:
        if isinstance(item, dict):
            yield from item["params"]
        else:
            yield item


def _refuse_bnb_state(optimizer, state_dict):
    for state in (state_dict.get("state") or {}).values():
        if isinstance(state, dict) and any(key in state for key in _BNB_STATE_KEYS):
            raise RuntimeError(
                "Unsloth: this optimizer checkpoint holds bitsandbytes 8-bit state, which the "
                "bitsandbytes stand-in's torch optimizer cannot resume. Resume with bitsandbytes "
                "installed, or start the optimizer fresh."
            )


def torch_optimizer_fallback(name, base):
    """A `base` subclass named `name` whose construction returns a plain `base` (torch) optimizer,
    ignoring bitsandbytes-only arguments (SGD / RMSprop / Adagrad: no 8-bit state here)."""

    def __new__(cls, params, *args, **kwargs):
        for key in _BNB_ONLY:
            kwargs.pop(key, None)
        params = _materialize([params] if isinstance(params, torch.Tensor) else params)
        if any(
            getattr(p, "dtype", None) is torch.float16 and p.requires_grad
            for p in _parameters(params)
        ):
            # bitsandbytes keeps fp32 math for fp16 weights; torch state in fp16 overflows.
            raise RuntimeError(
                f"Unsloth: `{name}` without bitsandbytes runs torch.optim.{base.__name__}, which "
                "cannot step float16 parameters safely. Train float32 / bfloat16 parameters, "
                "or install bitsandbytes."
            )
        if name not in _NOTICED:
            _NOTICED.add(name)
            # print, not warnings.warn: training scripts commonly silence warnings.
            print(
                f"Unsloth: bitsandbytes is unavailable, so `{name}` runs torch.optim."
                f"{base.__name__} with full precision optimizer states (more memory than 8-bit "
                "states, same update rule). Install bitsandbytes for 8-bit optimizers."
            )
        optimizer = base(params, *args, **kwargs)
        hook = getattr(optimizer, "register_load_state_dict_pre_hook", None)
        if hook is not None:
            hook(_refuse_bnb_state)
        return optimizer

    # A torch.optim subclass, so transformers >= 5 (which calls anything else a factory) builds it
    # like a class; __new__ hands back the plain `base` instance, so __init__ here never runs.
    return type(
        name,
        (base,),
        {"__new__": __new__, "__module__": "bitsandbytes.optim", "__qualname__": name},
    )
