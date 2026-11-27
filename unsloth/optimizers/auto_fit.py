# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Full finetuning that would run out of memory: step the optimizer inside backward.

A standard step keeps every gradient until optimizer.step(), so full finetuning needs the weights, a gradient
per weight and the optimizer state at once. When the planner predicts that this does not fit but the step
fits without the stored gradients, each weight is stepped by the user's own optimizer as soon as its gradient
is accumulated (a post-accumulate-grad hook), and the gradient is freed right away. AdamW / adamw_8bit / paged
optimizers keep their math, state, LR schedule, bfloat16 stochastic rounding and checkpoint format. Only
max_grad_norm changes: each tensor is clipped on its own (a global norm needs every gradient first). While the
global norm stays under max_grad_norm the result is bitwise the standard step.

When even that does not fit, LOMO (lomo-optim, plain SGD fused into backward) is used instead, which changes
the optimizer.

Engages only with gradient_accumulation_steps = 1 (more micro-batches must keep their gradients), full
finetuning, one process and no fp16 GradScaler. UNSLOTH_AUTO_FIT_OPTIMIZER = 0 turns it off; = in_backward,
lomo or adalomo forces a tier where it is eligible.

Also re-sizes lomo-optim AdaLomo's factored state for parameters with more than two dimensions (depthwise
convolutions, fused experts), which it sizes as if they were matrices and then fails on the first step.
"""

import contextlib
import os
import weakref

import torch

__all__ = [
    "auto_fit_mode",
    "InBackwardOptimizer",
    "install_auto_fit_optimizer",
    "plan_auto_fit",
    "fix_adalomo_state",
    "updates_in_backward",
]

_ENV = "UNSLOTH_AUTO_FIT_OPTIMIZER"
_OFF = ("0", "false", "no", "off")
_FORCED = ("in_backward", "lomo", "adalomo")
_MARK = "_unsloth_updates_in_backward"
_INSTALLED = "_unsloth_auto_fit_installed"


def auto_fit_mode():
    """'auto' (default), 'off', or a forced tier: 'in_backward', 'lomo', 'adalomo'."""
    value = os.environ.get(_ENV, "").strip().lower()
    if value in _OFF:
        return "off"
    if value in _FORCED:
        return value
    return "auto"


def updates_in_backward(trainer):
    """True when this trainer's optimizer changes the weights inside every backward (LOMO, AdaLomo, the
    *_layerwise optimizers, or the in-backward tier), so micro-batches cannot be split or regrouped."""
    if getattr(trainer, _MARK, False):
        return True
    optim = str(getattr(getattr(trainer, "args", None), "optim", "")).lower()
    return "lomo" in optim or "layerwise" in optim


def _unwrap(optimizer):
    seen = set()
    while optimizer is not None and id(optimizer) not in seen:
        seen.add(id(optimizer))
        inner = getattr(optimizer, "optimizer", None)
        if not isinstance(inner, torch.optim.Optimizer):
            return optimizer
        optimizer = inner
    return optimizer


def _bnb_kind(optimizer):
    try:
        from bitsandbytes.optim.optimizer import Optimizer8bit
    except Exception:
        return False
    return isinstance(optimizer, Optimizer8bit) and all(
        callable(getattr(optimizer, name, None))
        for name in ("check_overrides", "to_gpu", "init_state", "prefetch_state", "update_step")
    )


# torch optimizers whose step touches each parameter on its own (state keyed by the parameter, no
# cross-parameter reduction), so stepping one parameter's group is the same update.
_TORCH_PER_PARAM = (
    "Adam",
    "AdamW",
    "SGD",
    "Adagrad",
    "Adamax",
    "NAdam",
    "RAdam",
    "RMSprop",
    "Adafactor",
)


def _step_kind(optimizer):
    if _bnb_kind(optimizer):
        return "bitsandbytes"
    cls = type(optimizer)
    if cls.__module__.startswith("torch.optim") and cls.__name__ in _TORCH_PER_PARAM:
        return "torch"
    return None


@contextlib.contextmanager
def _no_global_step_hooks():
    """Global optimizer step hooks mean 'once per training step': the outer step fires them, not each weight."""
    import torch.optim.optimizer as module

    names = ("_global_optimizer_pre_hooks", "_global_optimizer_post_hooks")
    saved = {name: getattr(module, name, None) for name in names}
    try:
        for name, value in saved.items():
            if value is not None:
                setattr(module, name, type(value)())
        yield
    finally:
        for name, value in saved.items():
            if value is not None:
                setattr(module, name, value)


def _sync(p):
    try:
        from bitsandbytes.utils import sync_gpu
        sync_gpu(p)
    except Exception:
        if p.is_cuda:
            torch.cuda.synchronize(p.device)


_ACTIVE = weakref.WeakValueDictionary()  # id(param) -> the InBackwardOptimizer that owns its hook


class InBackwardOptimizer(torch.optim.Optimizer):
    """Steps `optimizer` one weight at a time from post-accumulate-grad hooks, then frees that gradient.

    Shares the wrapped optimizer's param_groups and state (the LR scheduler, bfloat16 stochastic rounding,
    bitsandbytes overrides and checkpoints see the optimizer the Trainer built). step() only steps weights whose
    gradient did not pass through a hook. `max_grad_norm` > 0 clips each gradient by its own norm."""

    def __init__(
        self,
        optimizer,
        max_grad_norm = 0.0,
    ):
        kind = _step_kind(optimizer)
        if kind is None:
            raise TypeError(f"{type(optimizer).__name__} cannot be stepped one weight at a time.")
        self.optimizer = optimizer
        self._kind = kind
        self.max_grad_norm = float(max_grad_norm or 0.0)
        self._armed = False
        self._done = set()
        self._norm_sq = {}
        self._handles = []
        self._where = None
        self._where_key = None
        self.hook_steps = 0
        self.late_steps = 0
        groups = list(optimizer.param_groups)
        state = optimizer.state
        # Optimizer.__init__ re-adds the same group dicts through the forwarding properties below.
        super().__init__(groups, optimizer.defaults)
        optimizer.state = state
        self._register_hooks()

    # Forwarded, so the wrapper and the wrapped optimizer can never disagree (load_state_dict replaces both).
    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optimizer.param_groups = value

    @property
    def state(self):
        return self.optimizer.state

    @state.setter
    def state(self, value):
        self.optimizer.state = value

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self._where = None
        return self.optimizer.load_state_dict(state_dict)

    def zero_grad(self, set_to_none = True):
        return self.optimizer.zero_grad(set_to_none = set_to_none)

    def _register_hooks(self):
        ref = weakref.ref(self)

        def hook(p):
            opt = ref()
            if opt is not None and opt._armed and _ACTIVE.get(id(p)) is opt:
                opt._step_param(p)

        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                previous = _ACTIVE.get(id(p))
                if previous is not None and previous is not self:
                    previous.close()
                _ACTIVE[id(p)] = self
                self._handles.append(p.register_post_accumulate_grad_hook(hook))

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        for key in [k for k, v in list(_ACTIVE.items()) if v is self]:
            _ACTIVE.pop(key, None)
        self._armed = False

    @contextlib.contextmanager
    def armed(self):
        previous = self._armed
        self._armed = True
        try:
            yield self
        finally:
            self._armed = previous

    def _locate(self, p):
        groups = self.optimizer.param_groups
        key = (id(groups), tuple(id(g) for g in groups), sum(len(g["params"]) for g in groups))
        if self._where is None or self._where_key != key:
            self._where = {
                id(q): (gi, pi) for gi, g in enumerate(groups) for pi, q in enumerate(g["params"])
            }
            self._where_key = key
        return self._where.get(id(p))

    def _step_param(self, p):
        grad = p.grad
        if grad is None:
            return
        where = self._locate(p)
        if where is None:
            return
        if id(p) in self._done:
            raise RuntimeError(
                "Unsloth: the optimizer steps inside backward (each gradient is used and freed as soon as it "
                "is ready), but a weight received a second gradient before optimizer.step(): this training "
                "step runs more than one backward (gradient accumulation, a split micro-batch, or a weight "
                "reused across checkpointed blocks). Set UNSLOTH_AUTO_FIT_OPTIMIZER=0 to keep every gradient "
                "until the step."
            )
        self._done.add(id(p))
        norm = torch.linalg.vector_norm(grad.detach(), 2.0, dtype = torch.float32)
        square = norm * norm
        device = str(norm.device)
        self._norm_sq[device] = (
            square if device not in self._norm_sq else self._norm_sq[device] + square
        )
        if self.max_grad_norm > 0:
            # torch.nn.utils.clip_grad_norm_'s coefficient, from this gradient's norm alone.
            coef = torch.clamp(self.max_grad_norm / (norm + 1e-6), max = 1.0)
            grad.detach().mul_(coef.to(grad.device))
        gi, pi = where
        with _no_global_step_hooks():
            if self._kind == "bitsandbytes":
                self._step_bitsandbytes(gi, pi, p)
            else:
                self._step_torch(gi, p)
        p.grad = None

    def _step_torch(self, gi, p):
        base = self.optimizer
        groups = base.param_groups
        try:
            base.param_groups = [dict(groups[gi], params = [p])]
            base.step()
        finally:
            base.param_groups = groups

    def _step_bitsandbytes(self, gi, pi, p):
        # bitsandbytes' own step body for one weight, with its real (group, index): GlobalOptimManager
        # overrides (32-bit embedding state) are keyed by those indices.
        base = self.optimizer
        if not getattr(base, "initialized", False):
            base.check_overrides()
            base.to_gpu()
            base.initialized = True
        group = base.param_groups[gi]
        state = base.state[p]
        if len(state) == 0:
            base.init_state(group, p, gi, pi)
        base.prefetch_state(p)
        base.update_step(group, p, gi, pi)
        _sync(p)

    def flush(self):
        """Step weights whose gradient reached .grad without a hook (assigned directly)."""
        for group in self.optimizer.param_groups:
            for p in list(group["params"]):
                if p.grad is not None and id(p) not in self._done:
                    self._step_param(p)
                    self.late_steps += 1
                elif p.grad is not None:
                    self._step_param(p)  # raises: a second gradient after the hook stepped it

    def total_norm(self):
        """L2 norm over every gradient this step saw (before clipping)."""
        squares = list(self._norm_sq.values())
        if not squares:
            return torch.tensor(0.0)
        device = squares[0].device
        return torch.stack([v.to(device) for v in squares]).sum().sqrt()

    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.flush()
        self.hook_steps += len(self._done)
        self._done = set()
        self._norm_sq = {}
        return loss


def _bitsandbytes_embedding_ids(model):
    ids = set()
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            ids.add(id(module.weight))
    return ids


def _state_bytes(optim_name, optimizer_cls, params, model):
    """GPU bytes the optimizer state takes for `params` (paged state can live in host RAM: counted as 0)."""
    name = (optim_name or "").lower()
    cls_name = getattr(optimizer_cls, "__name__", "") if optimizer_cls is not None else ""
    label = (name + " " + cls_name).lower()
    if "paged" in label:
        return 0
    if "lomo" in label or "sgd" in label or "adafactor" in label:
        return 0
    if "8bit" in label or "8_bit" in label:
        embedding = _bitsandbytes_embedding_ids(model) if model is not None else set()
        total = 0
        for p in params:
            n = p.numel()
            if id(p) in embedding or n < 4096:
                total += n * 8  # Trainer keeps embeddings' state in 32 bits; bnb min_8bit_size 4096
            else:
                total += n * 2 + (n // 256 + 1) * 8
        return total
    if "lion" in label:
        return sum(p.numel() * p.element_size() for p in params)
    if "bnb" in label or "32bit" in label:
        return sum(p.numel() * 8 for p in params)
    if "adam" in label:
        # torch Adam / AdamW keep exp_avg and exp_avg_sq in the parameter's dtype.
        return sum(p.numel() * p.element_size() * 2 for p in params)
    return sum(p.numel() * 8 for p in params)


def _seq_and_batch(trainer, model):
    args = getattr(trainer, "args", None)
    seq = None
    for key in ("max_length", "max_seq_length"):
        value = getattr(args, key, None)
        if isinstance(value, int) and value > 0:
            seq = value
            break
    if seq is None:
        seq = getattr(model, "max_seq_length", None) or 2048
    batch = int(getattr(args, "per_device_train_batch_size", 1) or 1)
    return int(seq), batch


def _reserve(model, seq, batch, extra):
    try:
        from unsloth_zoo.block_swap import estimate_training_reserve_bytes
    except Exception:
        estimate_training_reserve_bytes = None
    config = getattr(model, "config", None)
    if estimate_training_reserve_bytes is not None and config is not None:
        try:
            return int(
                estimate_training_reserve_bytes(config, seq, batch_size = batch, extra_bytes = extra)
            )
        except Exception:
            pass
    text = getattr(config, "text_config", None) or config
    hidden = int(getattr(text, "hidden_size", 0) or 0)
    vocab = int(getattr(text, "vocab_size", 0) or 0)
    tokens = max(1, seq) * max(1, batch)
    base = tokens * hidden * 48 + min(tokens, 2048) * vocab * 4 + extra
    return int(base + base / 16 + (256 << 20))


def _trainable(model):
    seen, out = set(), []
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


def plan_auto_fit(
    trainer,
    model,
    usable_bytes = None,
    optimizer_cls = None,
):
    """Memory prediction for one training step (bytes beyond what is allocated now).

    Returns a dict with `standard`, `in_backward`, `lomo` (bytes each needs), `usable`, and the parts."""
    args = getattr(trainer, "args", None)
    params = _trainable(model)
    grads = sum(p.numel() * p.element_size() for p in params)
    largest = max((p.numel() for p in params), default = 0)
    largest_bytes = max((p.numel() * p.element_size() for p in params), default = 0)
    optim_name = str(getattr(args, "optim", "") or "")
    custom = getattr(trainer, "optimizer_cls_and_kwargs", None)
    if optimizer_cls is None and custom is not None:
        optimizer_cls = custom[0]
        optim_name = ""
    states = _state_bytes(optim_name, optimizer_cls, params, model)
    # Standard step transients: torch's foreach AdamW holds a parameter-sized temporary per tensor list,
    # bitsandbytes / the bfloat16 one-pass kernel only per weight.
    label = (optim_name + " " + getattr(optimizer_cls, "__name__", "")).lower()
    per_weight = "8bit" in label or "bnb" in label or "paged" in label
    step_temp = largest * 8 if per_weight else grads
    # In backward: the gradient being stepped, one more still accumulating (a tied embedding's head
    # gradient lives the whole backward), and a float32 copy of one weight.
    live = largest_bytes * 2 + largest * 8
    seq, batch = _seq_and_batch(trainer, model)
    standard = _reserve(model, seq, batch, grads + states + step_temp)
    in_backward = _reserve(model, seq, batch, states + live)
    lomo = _reserve(model, seq, batch, live)
    if usable_bytes is None:
        usable_bytes = _usable_bytes(params)
    return dict(
        usable = usable_bytes,
        standard = standard,
        in_backward = in_backward,
        lomo = lomo,
        grads = grads,
        states = states,
        params = sum(p.numel() for p in params),
        seq = seq,
        batch = batch,
    )


def _usable_bytes(params):
    device = next((p.device for p in params if p.device.type == "cuda"), None)
    if device is None:
        return None
    try:
        from ..models._utils import usable_cuda_bytes
        return usable_cuda_bytes(device)
    except Exception:
        free, _ = torch.cuda.mem_get_info(device)
        return int(free)


def _single_process(trainer):
    args = getattr(trainer, "args", None)
    if int(getattr(args, "world_size", 1) or 1) > 1 or int(getattr(args, "n_gpu", 1) or 1) > 1:
        return False
    if getattr(trainer, "is_deepspeed_enabled", False) or getattr(
        trainer, "is_fsdp_enabled", False
    ):
        return False
    try:
        from ..models.loader_utils import is_distributed
        if is_distributed():
            return False
    except Exception:
        pass
    dist = getattr(torch, "distributed", None)
    try:
        if dist is not None and dist.is_available() and dist.is_initialized():
            return dist.get_world_size() <= 1
    except Exception:
        return False
    return True


def _full_finetuning(model):
    seen = set()
    while model is not None and id(model) not in seen:
        seen.add(id(model))
        flag = getattr(model, "_unsloth_full_finetuning", None)
        if flag is not None:
            return bool(flag)
        model = getattr(model, "module", None) or getattr(model, "base_model", None)
    return False


def _blocker(trainer, model, forced):
    """Why neither tier may engage for this trainer (None: eligible)."""
    args = getattr(trainer, "args", None)
    if int(getattr(args, "gradient_accumulation_steps", 1) or 1) != 1:
        return "gradient_accumulation_steps > 1 (every micro-batch's gradient must be kept until the step)"
    if not _single_process(trainer):
        return "distributed training (DDP / FSDP / DeepSpeed) reduces gradients after backward"
    if getattr(args, "fp16", False):
        return "fp16 mixed precision (the GradScaler skips whole steps on overflow)"
    if not forced and not _full_finetuning(model):
        return "not full finetuning"
    if getattr(model, "_unsloth_block_swap", None) is not None:
        return "block swap (offload_layers) keeps the weights in host RAM"
    if os.environ.get("UNSLOTH_OFFLOAD_OPTIMIZER_STATE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return "UNSLOTH_OFFLOAD_OPTIMIZER_STATE"
    if getattr(args, "q_galore_config", None) is not None:
        return "Q-GaLore"
    optim = str(getattr(args, "optim", "")).lower()
    if any(
        k in optim for k in ("galore", "apollo", "layerwise", "lomo", "schedule_free", "torchao")
    ):
        return f"optim = {optim}"
    if "_unsloth_token_budget_splits" in getattr(trainer, "__dict__", {}):
        return "UNSLOTH_TOKEN_BUDGET splits micro-batches"
    return None


def _gb(x):
    return f"{x / 2**30:.1f} GB"


def _lomo_available():
    try:
        import lomo_optim  # noqa: F401
        return True
    except Exception:
        return False


def _decide(trainer, model, mode):
    """(tier, message): tier is None, 'in_backward', 'lomo' or 'adalomo'."""
    forced = mode in _FORCED
    blocker = _blocker(trainer, model, forced)
    args = getattr(trainer, "args", None)
    if blocker is not None:
        if not forced:
            if not _full_finetuning(model) or not str(blocker).startswith("gradient_accumulation"):
                return None, None
            plan = plan_auto_fit(trainer, model)
            if plan["usable"] is None or plan["standard"] <= plan["usable"]:
                return None, None
            return None, (
                f"Unsloth: full finetuning is predicted to run out of memory ({_gb(plan['standard'])} needed for "
                f"gradients, optimizer state and activations, {_gb(plan['usable'])} free). With "
                f"gradient_accumulation_steps = 1 (raise per_device_train_batch_size instead) the optimizer can step "
                f"inside backward and never hold every gradient at once."
            )
        return None, f"Unsloth: {_ENV}={mode} ignored: {blocker}."
    if mode == "in_backward":
        return "in_backward", f"Unsloth: the optimizer steps inside backward ({_ENV}=in_backward)."
    if mode in ("lomo", "adalomo"):
        if not _lomo_available():
            return None, f"Unsloth: {_ENV}={mode} needs `pip install lomo-optim`."
        if getattr(trainer, "optimizer_cls_and_kwargs", None) is not None:
            return (
                None,
                f"Unsloth: {_ENV}={mode} ignored: the trainer was given its own optimizer class.",
            )
        return (
            mode,
            f"Unsloth: using {mode.upper() if mode == 'lomo' else 'AdaLomo'} ({_ENV}={mode}).",
        )
    plan = plan_auto_fit(trainer, model)
    if plan["usable"] is None or plan["standard"] <= plan["usable"]:
        return None, None
    optim = str(getattr(args, "optim", "")) or "the optimizer"
    head = (
        f"Unsloth: full finetuning of {plan['params'] / 1e9:.2f}B weights is predicted to run out of memory: "
        f"{_gb(plan['standard'])} needed for gradients ({_gb(plan['grads'])}), {optim} state "
        f"({_gb(plan['states'])}) and activations, {_gb(plan['usable'])} free."
    )
    if plan["in_backward"] <= plan["usable"]:
        clip = (
            f" max_grad_norm = {args.max_grad_norm} now clips each tensor by its own norm."
            if (getattr(args, "max_grad_norm", 0) or 0) > 0
            else ""
        )
        return "in_backward", (
            f"{head} So {optim} steps each weight inside backward and frees its gradient at once "
            f"(predicted {_gb(plan['in_backward'])}); the optimizer math is unchanged.{clip} "
            f"Set {_ENV}=0 to turn this off."
        )
    if plan["lomo"] <= plan["usable"]:
        if not _lomo_available():
            return None, (
                f"{head} Even stepping inside backward needs {_gb(plan['in_backward'])} because of the optimizer "
                f"state. LOMO would fit ({_gb(plan['lomo'])}): `pip install lomo-optim`."
            )
        if getattr(trainer, "optimizer_cls_and_kwargs", None) is not None:
            return (
                None,
                f"{head} LOMO would fit, but the trainer was given its own optimizer class.",
            )
        return "lomo", (
            f"{head} Even stepping {optim} inside backward needs {_gb(plan['in_backward'])} for its state, so "
            f"training uses LOMO (plain SGD fused into backward, no optimizer state, predicted "
            f"{_gb(plan['lomo'])}). This changes the optimizer: SGD usually needs a much larger learning_rate "
            f"than AdamW, and max_grad_norm is not applied. Set {_ENV}=0 to turn this off."
        )
    return None, (
        f"{head} Neither stepping inside backward ({_gb(plan['in_backward'])}) nor LOMO ({_gb(plan['lomo'])}) "
        f"fits; consider LoRA, a shorter max_length or freezing part of the model."
    )


def fix_adalomo_state(optimizer):
    """lomo-optim AdaLomo sizes its factored second moment as rows = shape[0], cols = shape[1] for any weight
    with more than one dimension, on the default CUDA device. A 3-D weight (a depthwise Conv1d [C, 1, k], fused
    experts [E, out, in]) then fails on its first update with `output with shape [C] doesn't match the
    broadcast shape [C, C]`. Re-size it the way Adafactor factors (rows shape[:-1], cols shape[:-2] + shape[-1:]),
    on the weight's device; 2-D weights keep exactly their state. Returns the number of tensors re-sized."""
    optimizer = _unwrap(optimizer)
    if type(optimizer).__name__ != "AdaLomo" or getattr(optimizer, "zero3_enabled", False):
        return 0
    rows = getattr(optimizer, "exp_avg_sq_row", None)
    cols = getattr(optimizer, "exp_avg_sq_col", None)
    full = getattr(optimizer, "exp_avg_sq", None)
    model = getattr(optimizer, "model", None)
    if not isinstance(rows, dict) or not isinstance(cols, dict) or model is None:
        return 0
    fixed = 0
    for name, p in model.named_parameters():
        shape = tuple(p.shape)
        if len(shape) > 1:
            for table, want in ((rows, shape[:-1]), (cols, shape[:-2] + shape[-1:])):
                value = table.get(name)
                if value is None or tuple(value.shape) != want or value.device != p.device:
                    table[name] = torch.zeros(want, dtype = torch.float32, device = p.device)
                    fixed += 1
        elif len(shape) == 1 and isinstance(full, dict):
            value = full.get(name)
            if value is None or value.device != p.device:
                full[name] = torch.zeros(shape, dtype = torch.float32, device = p.device)
                fixed += 1
    return fixed


def _set_optim(args, name):
    try:
        from transformers.training_args import OptimizerNames
        value = OptimizerNames(name)
    except Exception:
        value = name
    try:
        args.optim = value
    except Exception:
        object.__setattr__(args, "optim", value)


def _wrap_clip(trainer):
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is None or getattr(accelerator.clip_grad_norm_, "_unsloth_in_backward", False):
        return
    original = accelerator.clip_grad_norm_
    trainer_ref = weakref.ref(trainer)

    def clip_grad_norm_(
        parameters,
        max_norm,
        norm_type = 2,
        *args,
        **kwargs,
    ):
        trainer = trainer_ref()
        wrapper = getattr(trainer, "_unsloth_in_backward_optimizer", None)
        if wrapper is None or float(norm_type) != 2.0:
            return original(parameters, max_norm, norm_type, *args, **kwargs)
        # The weights were already stepped (each clipped by its own norm); report the step's global norm.
        wrapper.flush()
        return wrapper.total_norm()

    clip_grad_norm_._unsloth_in_backward = True
    clip_grad_norm_.__wrapped__ = original
    accelerator.clip_grad_norm_ = clip_grad_norm_


def _arm_training_step(trainer, wrapper):
    original = trainer.training_step
    if getattr(original, "_unsloth_in_backward", False):
        original = original.__wrapped__
    wrapper_ref = weakref.ref(wrapper)

    def training_step(*args, **kwargs):
        opt = wrapper_ref()
        if opt is None:
            return original(*args, **kwargs)
        with opt.armed():
            return original(*args, **kwargs)

    training_step._unsloth_in_backward = True
    training_step.__wrapped__ = original
    trainer.training_step = training_step


def _convert(trainer, optimizer, message):
    base = _unwrap(optimizer)
    if isinstance(base, InBackwardOptimizer) or _step_kind(base) is None:
        name = type(base).__name__
        print(f"Unsloth: not stepping {name} inside backward: it is not a per-weight optimizer.")
        return optimizer
    max_norm = float(getattr(trainer.args, "max_grad_norm", 0) or 0)
    wrapper = InBackwardOptimizer(base, max_grad_norm = max_norm)
    setattr(trainer, _MARK, True)
    trainer._unsloth_in_backward_optimizer = wrapper
    _wrap_clip(trainer)
    _arm_training_step(trainer, wrapper)
    if message:
        print(message)
    return wrapper


def install_auto_fit_optimizer(trainer):
    """Wrap this trainer's create_optimizer (idempotent; transformers 4.x reaches it through
    create_optimizer_and_scheduler, 5.x calls it directly). Returns True when installed."""
    if getattr(trainer, _INSTALLED, False):
        return True
    original = getattr(trainer, "create_optimizer", None)
    if original is None:
        return False

    def create_optimizer(*args, **kwargs):
        model = getattr(trainer, "model", None)
        if getattr(trainer, "optimizer", None) is not None or model is None:
            result = original(*args, **kwargs)
            fix_adalomo_state(getattr(trainer, "optimizer", None) or result)
            return result
        mode = auto_fit_mode()
        tier, message = (None, None) if mode == "off" else _decide(trainer, model, mode)
        if tier in ("lomo", "adalomo"):
            _set_optim(trainer.args, tier)
            setattr(trainer, _MARK, True)
        if message and tier != "in_backward":
            print(message)
        result = original(*args, **kwargs)
        optimizer = getattr(trainer, "optimizer", None) or result
        fix_adalomo_state(optimizer)
        if tier != "in_backward" or optimizer is None:
            return result
        trainer.optimizer = _convert(trainer, optimizer, message)
        return trainer.optimizer

    create_optimizer.__wrapped__ = original
    trainer.create_optimizer = create_optimizer
    setattr(trainer, _INSTALLED, True)
    return True
