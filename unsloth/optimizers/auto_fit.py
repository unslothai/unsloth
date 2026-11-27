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

When even that does not fit, LOMO's update is used instead: plain SGD stepped the same way, with no optimizer
state. This changes the optimizer.

Engages only with gradient_accumulation_steps = 1 (more micro-batches must keep their gradients), full
finetuning, one process and no fp16 GradScaler. UNSLOTH_AUTO_FIT_OPTIMIZER = 0 turns it off; = in_backward or
= lomo forces a tier where it is eligible, = adalomo uses lomo-optim's AdaLomo.

Also re-sizes lomo-optim AdaLomo's factored state for parameters with more than two dimensions (depthwise
convolutions, fused experts), which it sizes as if they were matrices and then fails on the first step.
"""

import contextlib
import json
import os
import sys
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
_TIER = "_unsloth_auto_fit_tier"
_TIER_FILE = "unsloth_auto_fit.json"  # in each checkpoint: the optimizer tier it was trained with


def auto_fit_mode():
    """'auto' (default), 'off', or a forced tier: 'in_backward', 'lomo', 'adalomo'."""
    value = os.environ.get(_ENV, "").strip().lower()
    if value in _OFF:
        return "off"
    if value in _FORCED:
        return value
    return "auto"


def _optim_name(args):
    """TrainingArguments.optim as its string value (an OptimizerNames enum prints as `OptimizerNames.ADAMW_BNB`)."""
    optim = getattr(args, "optim", "") or ""
    return str(getattr(optim, "value", optim))


def updates_in_backward(trainer):
    """True when this trainer's optimizer changes the weights inside every backward (LOMO, AdaLomo, the
    *_layerwise optimizers, or the in-backward tier), so micro-batches cannot be split or regrouped."""
    if getattr(trainer, _MARK, False):
        return True
    optim = _optim_name(getattr(trainer, "args", None)).lower()
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
    # transformers' Adafactor (optim="adafactor") factors each weight on its own as well.
    if cls.__module__.startswith("transformers.") and cls.__name__ == "Adafactor":
        return "torch"
    return None


def _graph_task():
    graph_task = getattr(torch._C, "_current_graph_task_id", None)
    return graph_task() if graph_task is not None else -1


def _in_backward_pass():
    return _graph_task() != -1


def _compiling():
    # Constant True while dynamo traces: a hook inlined into a compiled region adds no graph break.
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if is_compiling is None:
        is_compiling = getattr(getattr(torch, "_dynamo", None), "is_compiling", None)
    return bool(is_compiling()) if is_compiling is not None else False


def _in_block_trial():
    """Unsloth's compiled decoder blocks run one throwaway grad-mode forward on their first call (the trial that
    checks the compiled region); it makes no gradient, so it is not a second use of the block's weights."""
    module = sys.modules.get("unsloth.models._block_compile")
    flag = getattr(module, "_IN_TRIAL", None) if module is not None else None
    return bool(flag[0]) if isinstance(flag, list) and flag else False


def _norms(grads):
    """float32 L2 norm of each gradient, in one multi-tensor launch where torch has it."""
    foreach_norm = getattr(torch, "_foreach_norm", None)
    if foreach_norm is not None:
        try:
            return list(foreach_norm(grads, 2.0, dtype = torch.float32))
        except (TypeError, RuntimeError):
            pass
    return [torch.linalg.vector_norm(g, 2.0, dtype = torch.float32) for g in grads]


def _queue_callback(callback):
    """Run `callback` when the current backward graph task ends (a reentrant checkpoint's inner backward is its
    own task, so with Unsloth's checkpointing that is once per decoder layer). False when it cannot be queued."""
    engine = getattr(getattr(torch.autograd, "Variable", None), "_execution_engine", None)
    queue = getattr(engine, "queue_callback", None)
    if queue is None:
        return False
    try:
        queue(callback)
        return True
    except RuntimeError:
        return False


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
    """Steps `optimizer` inside backward from post-accumulate-grad hooks, then frees those gradients.

    Ready gradients are stepped together when the backward graph task that produced them ends (with reentrant
    checkpointing, after each block's backward) or once they reach `batch_bytes`, so the optimizer, the norms and
    the clipping run as a few multi-tensor launches per block instead of several per weight; each weight's update
    is the one the standard step makes. Shares the wrapped optimizer's param_groups and state (the LR scheduler,
    bfloat16 stochastic rounding, bitsandbytes overrides and checkpoints see the optimizer the Trainer built).
    step() only steps weights whose gradient did not pass through a hook. `max_grad_norm` > 0 clips each gradient
    by its own norm. `batch_bytes` = 0 steps each weight on its own as soon as its gradient is ready."""

    def __init__(
        self,
        optimizer,
        max_grad_norm = 0.0,
        model = None,
        batch_bytes = None,
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
        self.deferred_steps = 0
        self._calls = {}
        self._owners = {}
        self._shared = set()
        self._pending = []
        self._pending_ids = set()
        self._pending_bytes = 0
        self._queued = set()
        groups = list(optimizer.param_groups)
        state = optimizer.state
        # Optimizer.__init__ re-adds the same group dicts through the forwarding properties below.
        super().__init__(groups, optimizer.defaults)
        optimizer.state = state
        if batch_bytes is None:
            # The largest weight's gradient (the planner budgets it alive next to the one being stepped).
            batch_bytes = max(
                (p.numel() * p.element_size() for g in groups for p in g["params"]), default = 0
            )
        self.batch_bytes = int(batch_bytes)
        self._register_hooks()
        if model is not None:
            self._watch_reuse(model)

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
        if isinstance(self.optimizer, torch.optim.SGD) and isinstance(state_dict, dict):
            # A checkpoint of another optimizer (AdamW): SGD without momentum keeps no state, so do not load
            # its moments onto the GPU, and keep SGD's own group options (momentum, dampening, ...), taking only
            # the saved learning rate and weight decay.
            keep = {"momentum_buffer"}
            state = {
                k: {n: v for n, v in s.items() if n in keep}
                for k, s in (state_dict.get("state") or {}).items()
            }
            saved = state_dict.get("param_groups") or []
            groups = []
            for current, group in zip(self.optimizer.param_groups, saved):
                merged = {k: v for k, v in current.items() if k != "params"}
                merged.update(
                    {k: group[k] for k in ("lr", "weight_decay", "initial_lr") if k in group}
                )
                merged["params"] = group["params"]
                groups.append(merged)
            if len(groups) == len(saved):
                state_dict = dict(state_dict, param_groups = groups)
            state_dict = dict(state_dict, state = {k: s for k, s in state.items() if s})
        elif isinstance(state_dict, dict) and state_dict.get("param_groups"):
            # A checkpoint of the SGD tier lacks this optimizer's group options (betas, eps): keep ours for those.
            saved = state_dict["param_groups"]
            if len(saved) == len(self.optimizer.param_groups):
                state_dict = dict(
                    state_dict,
                    param_groups = [
                        {**{k: v for k, v in current.items() if k != "params"}, **group}
                        for current, group in zip(self.optimizer.param_groups, saved)
                    ],
                )
        return self.optimizer.load_state_dict(state_dict)

    def _reset(self):
        self._done = set()
        self._norm_sq = {}
        self._calls = {}
        self._pending = []
        self._pending_ids = set()
        self._pending_bytes = 0
        self._queued = set()

    def zero_grad(self, set_to_none = True):
        self._reset()
        return self.optimizer.zero_grad(set_to_none = set_to_none)

    def _register_hooks(self):
        ref = weakref.ref(self)

        def hook(p):
            opt = ref()
            if opt is not None and opt._armed and _ACTIVE.get(id(p)) is opt:
                opt._ready(p)

        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                previous = _ACTIVE.get(id(p))
                if previous is not None and previous is not self:
                    previous.close()
                _ACTIVE[id(p)] = self
                self._handles.append(p.register_post_accumulate_grad_hook(hook))

    def _watch_reuse(self, model):
        """Count module calls during the forward (not the checkpoint recompute inside backward). A weight is reused
        when a module that owns it ran more than once in total (a tied embedding and head), or any module containing
        it ran more than once (a looped block, even when the weight is read functionally, as Inkling-2's loop
        convolutions are). Under reentrant checkpointing such a weight gets one gradient per call, so it is stepped
        at optimizer.step() instead. A weight the model reaches through several modules (tied, shared) always is.

        Calls inside a compiled region are not counted (the hook compiles to nothing, so it does not split the
        graph; they run within one call of an eager parent, which is counted), nor are Unsloth's first-call block
        trials, which make no gradient."""
        mine = {id(p) for group in self.param_groups for p in group["params"]}
        self._names = {id(p): n for n, p in model.named_parameters() if id(p) in mine}
        paths = {}
        for name, module in model.named_modules(remove_duplicate = False):
            for p in module.parameters(recurse = False):
                if id(p) in mine:
                    paths.setdefault(id(p), set()).add(name)
        self._shared = {k for k, v in paths.items() if len(v) > 1}
        ref = weakref.ref(self)

        def count(module, args):
            if _compiling():
                return
            opt = ref()
            if opt is not None and opt._armed and not _in_backward_pass() and not _in_block_trial():
                opt._calls[id(module)] = opt._calls.get(id(module), 0) + 1

        owners, within = {}, {}

        def walk(module, ancestors):
            here = ancestors + (id(module),)
            holds = False
            for p in module.parameters(recurse = False):
                if id(p) in mine:
                    owners.setdefault(id(p), []).append(id(module))
                    within.setdefault(id(p), set()).update(here)
                    holds = True
            for child in module.children():
                holds = walk(child, here) or holds
            if holds:
                self._handles.append(module.register_forward_pre_hook(count))
            return holds

        walk(model, ())
        self._owners = {k: (tuple(v), tuple(within[k])) for k, v in owners.items()}

    def _reused(self, p):
        if id(p) in self._shared:
            return True
        entry = self._owners.get(id(p))
        if not entry:
            return False
        direct, containing = entry
        calls = self._calls
        if sum(calls.get(m, 0) for m in direct) > 1:
            return True
        return any(calls.get(m, 0) > 1 for m in containing)

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
        except BaseException:
            # The weights stepped so far stay stepped; the next step starts clean.
            self._reset()
            raise
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

    def _second_gradient(self, p):
        name = getattr(self, "_names", {}).get(id(p), "a weight")
        return RuntimeError(
            "Unsloth: the optimizer steps inside backward (each gradient is used and freed as soon as it "
            f"is ready), but {name} received a second gradient before optimizer.step(): this training "
            "step runs more than one backward (gradient accumulation, a split micro-batch, or a weight "
            "reused across checkpointed blocks). Set UNSLOTH_AUTO_FIT_OPTIMIZER=0 to keep every gradient "
            "until the step."
        )

    def _ready(self, p):
        """p's gradient is accumulated: queue it for the batch this graph task steps."""
        grad = p.grad
        if grad is None or self._locate(p) is None:
            return
        if self._reused(p):
            return  # more gradients may still arrive: stepped by flush()
        if id(p) in self._done:
            raise self._second_gradient(p)
        if id(p) in self._pending_ids:
            return  # accumulated again before its batch was stepped: stepped once, with the sum
        nbytes = grad.numel() * grad.element_size()
        if self._pending and self._pending_bytes + nbytes > self.batch_bytes:
            self._flush_pending()  # a batch never exceeds batch_bytes (one larger weight steps alone)
        self._pending.append(p)
        self._pending_ids.add(id(p))
        self._pending_bytes += nbytes
        if self._pending_bytes >= self.batch_bytes or not self._queue_flush():
            self._flush_pending()

    def _queue_flush(self):
        task = _graph_task()
        if task == -1:
            return False
        if task in self._queued:
            return True
        ref = weakref.ref(self)

        def flush_at_end_of_task():
            opt = ref()
            if opt is not None:
                opt._queued.discard(task)
                opt._flush_pending()

        if not _queue_callback(flush_at_end_of_task):
            return False
        self._queued.add(task)
        return True

    def _flush_pending(self):
        pending = self._pending
        if not pending:
            return
        self._pending = []
        self._pending_ids = set()
        self._pending_bytes = 0
        self._step_params(pending)

    def _step_params(self, params):
        """Step these weights with their gradients now (each by its own update), then free the gradients."""
        params = [p for p in params if p.grad is not None and self._locate(p) is not None]
        if not params:
            return
        for p in params:
            if id(p) in self._done:
                raise self._second_gradient(p)
        for p in params:
            self._done.add(id(p))
        by_device = {}
        for p in params:
            by_device.setdefault(p.grad.device, []).append(p)
        for device, group in by_device.items():
            grads = [p.grad.detach() for p in group]
            norms = torch.stack(_norms(grads))
            square = (norms * norms).sum()
            key = str(norms.device)
            self._norm_sq[key] = square if key not in self._norm_sq else self._norm_sq[key] + square
            if self.max_grad_norm > 0:
                # torch.nn.utils.clip_grad_norm_'s coefficient, from each gradient's own norm.
                coef = torch.clamp(self.max_grad_norm / (norms + 1e-6), max = 1.0)
                torch._foreach_mul_(grads, list(coef.unbind(0)))
        with _no_global_step_hooks():
            if self._kind == "bitsandbytes":
                for p in params:
                    gi, pi = self._locate(p)
                    self._step_bitsandbytes(gi, pi, p)
            else:
                self._step_torch(params)
        for p in params:
            p.grad = None

    def _step_torch(self, params):
        base = self.optimizer
        groups = base.param_groups
        by_group = {}
        for p in params:
            by_group.setdefault(self._locate(p)[0], []).append(p)
        try:
            for gi, batch in by_group.items():
                base.param_groups = [dict(groups[gi], params = batch)]
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
        # bitsandbytes' step syncs the device after every weight. Inside backward that would drain the queued
        # backward kernels once per weight; its kernels run on the current stream, so only paged state (async
        # page prefetches) needs it.
        if getattr(base, "is_paged", False):
            _sync(p)

    def flush(self):
        """Step the weights left for the end: the last batch, reused ones, and gradients that reached .grad
        without a hook."""
        self._flush_pending()
        left = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if id(p) not in self._done:
                    if self._reused(p):
                        self.deferred_steps += 1
                    else:
                        self.late_steps += 1
                # A weight the hook already stepped raises in _step_params: it received a second gradient.
                left.append(p)
        # In batches of at most batch_bytes, each freed before the next: a multi-tensor step's temporaries span
        # its whole batch, and the planner budgets them for one batch.
        batch, size = [], 0
        for p in left:
            nbytes = p.grad.numel() * p.grad.element_size()
            if batch and size + nbytes > self.batch_bytes:
                self._step_params(batch)
                batch, size = [], 0
            batch.append(p)
            size += nbytes
        self._step_params(batch)

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
        self._reset()
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
        # Paged state lives in managed memory; bitsandbytes keeps tensors under 1e5 elements on the device.
        return sum(p.numel() * 8 for p in params if p.numel() < 1e5)
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


def _activation_bytes(model, seq, batch):
    """Checkpointed activations + float32 logits rows, as unsloth_zoo.block_swap.estimate_training_reserve_bytes
    counts them (its margins are added once in plan_auto_fit)."""
    config = getattr(model, "config", None)
    try:
        from unsloth_zoo.block_swap import (
            _ACTIVATION_BYTES_PER_TOKEN_HIDDEN as per_token,
            _text_config,
        )
        text = _text_config(config)
    except Exception:
        per_token = 48
        text = getattr(config, "text_config", None) or config
    hidden = int(
        getattr(text, "hidden_size", None)
        or getattr(text, "n_embd", None)
        or getattr(text, "d_model", None)
        or 0
    )
    vocab = int(getattr(text, "vocab_size", None) or 0)
    tokens = max(1, int(seq)) * max(1, int(batch))
    return tokens * hidden * int(per_token) + min(tokens, 2048) * vocab * 4


def _loop_layer_ids(model):
    """Decoder layers a looped model (a text config with loop_config.num_passes > 1, as Inkling-2) runs again."""
    config = getattr(model, "config", None)
    text = getattr(config, "text_config", None) or config
    loop = getattr(text, "loop_config", None)
    if isinstance(loop, dict):
        passes, start, end = loop.get("num_passes", 1), loop.get("core_start"), loop.get("core_end")
    else:
        passes = getattr(loop, "num_passes", 1)
        start, end = getattr(loop, "core_start", None), getattr(loop, "core_end", None)
    try:
        if int(passes or 1) <= 1 or start is None or end is None:
            return set()
        return set(range(int(start), int(end)))
    except (TypeError, ValueError):
        return set()


def _reused_bytes(model, params):
    """Gradients the in-backward step keeps until optimizer.step() that the model's structure shows: weights shared
    by several modules or several paths (tied embeddings, shared blocks) and a looped model's loop layers. Reuse
    that only the forward shows is deferred as well, but is not budgeted here."""
    mine = {id(p): p for p in params}
    reused = {}
    seen = {}
    for name, module in model.named_modules(remove_duplicate = False):
        for p in module.parameters(recurse = False):
            if id(p) in mine:
                seen.setdefault(id(p), set()).add(name)
    for pid, names in seen.items():
        if len(names) > 1:
            reused[pid] = mine[pid]
    loops = _loop_layer_ids(model)
    if loops:
        import re
        for name, p in model.named_parameters():
            m = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if m and int(m.group(1)) in loops and id(p) in mine:
                reused[id(p)] = p
    return sum(p.numel() * p.element_size() for p in reused.values())


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
    optim_name = _optim_name(args)
    custom = getattr(trainer, "optimizer_cls_and_kwargs", None)
    if optimizer_cls is None and custom is not None:
        optimizer_cls = custom[0]
        optim_name = ""
    states = _state_bytes(optim_name, optimizer_cls, params, model)
    label = (optim_name + " " + getattr(optimizer_cls, "__name__", "")).lower()
    # Measured on Llama-3.2-3B and Qwen3-0.6B / Qwen3-4B full finetuning (adamw_8bit, Unsloth checkpointing):
    # the standard step peaks at gradients + state + about the activations. torch's foreach AdamW (without the
    # bfloat16 one-pass kernel) also holds a moment-sized temporary over the whole tensor list.
    per_weight = (
        "8bit" in label or "bnb" in label or "paged" in label or "fused" in label or _sr_on(params)
    )
    step_temp = 0 if per_weight else states // 2
    # In backward: the batch of ready gradients being stepped (at most the largest weight's size) and the
    # gradient that arrived while it waited. A multi-tensor update without a one-pass kernel (torch foreach
    # AdamW on float32 weights, or with stochastic rounding off) also holds a moment-sized temporary over that
    # batch. The largest weight (the embedding) is stepped last, once activations are freed, so it overlaps them.
    # The SGD tier (LOMO's update) keeps no state but steps a float32 copy of a weight and its gradient.
    live = largest_bytes * 2 + (0 if per_weight else largest_bytes)
    live_lomo = largest_bytes * 2 + largest * 8
    seq, batch = _seq_and_batch(trainer, model)
    activations = _activation_bytes(model, seq, batch)
    margin = activations // 16 + (256 << 20)  # split allocator blocks + a fixed safety margin
    # Reused weights (tied, shared, looped) keep their gradient until the step in both in-backward tiers.
    deferred = _reused_bytes(model, params)
    standard = grads + states + max(activations, step_temp) + margin
    in_backward = states + deferred + max(activations, live) + margin
    lomo = deferred + max(activations, live_lomo) + margin
    if usable_bytes is None:
        usable_bytes = _usable_bytes(params)
    return dict(
        usable = usable_bytes,
        standard = standard,
        in_backward = in_backward,
        lomo = lomo,
        grads = grads,
        states = states,
        deferred = deferred,
        params = sum(p.numel() for p in params),
        seq = seq,
        batch = batch,
    )


def _sr_on(params):
    if os.environ.get("UNSLOTH_BF16_STOCHASTIC_ROUNDING", "1").strip().lower() in _OFF:
        return False
    return any(p.dtype == torch.bfloat16 for p in params)


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
    if (
        getattr(trainer, "is_fsdp_xla_enabled", False)
        or getattr(args, "fsdp", None)
        or getattr(args, "deepspeed", None)
    ):
        return False
    try:
        # FSDP / DeepSpeed plugins and Unsloth's native FSDP2, as the fused 8-bit AdamW checks them.
        from .flat_adamw8bit import _distributed_wrapper
        if _distributed_wrapper(trainer):
            return False
    except Exception:
        pass
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
        model = (
            getattr(model, "module", None)
            or getattr(model, "_orig_mod", None)
            or getattr(model, "base_model", None)
        )
    return False


def _scaler_active(trainer):
    accelerator = getattr(trainer, "accelerator", None)
    if getattr(accelerator, "scaler", None) is not None:
        return True
    return str(getattr(accelerator, "mixed_precision", "") or "").lower() == "fp16"


def _blocker(trainer, model, forced):
    """Why neither tier may engage for this trainer (None: eligible)."""
    args = getattr(trainer, "args", None)
    if int(getattr(args, "gradient_accumulation_steps", 1) or 1) != 1:
        return "gradient_accumulation_steps > 1 (every micro-batch's gradient must be kept until the step)"
    if not _single_process(trainer):
        return "distributed training (DDP / FSDP / DeepSpeed) reduces gradients after backward"
    if getattr(args, "fp16", False) or _scaler_active(trainer):
        return "fp16 mixed precision (the GradScaler skips whole steps on overflow)"
    if not forced and not _full_finetuning(model):
        return "not full finetuning"
    if getattr(args, "auto_find_batch_size", False):
        return "auto_find_batch_size (a retry after running out of memory would start from partly stepped weights)"
    if getattr(args, "enable_jit_checkpoint", False):
        return (
            "enable_jit_checkpoint (a checkpoint taken mid-step would hold partly stepped weights)"
        )
    if getattr(args, "torch_compile", False):
        return "torch_compile over the whole model (its backward returns every gradient at once)"
    if any(getattr(m, "_unsloth_block_swap", None) is not None for m in model.modules()):
        return "block swap (offload_layers) keeps the weights in host RAM"
    devices = {p.device for p in _trainable(model)}
    if len(devices) > 1:
        return "the model is split across devices"
    if os.environ.get("UNSLOTH_OFFLOAD_OPTIMIZER_STATE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return "UNSLOTH_OFFLOAD_OPTIMIZER_STATE"
    if getattr(args, "q_galore_config", None) is not None:
        return "Q-GaLore"
    optim = _optim_name(args).lower()
    if any(
        k in optim
        for k in (
            "galore",
            "apollo",
            "layerwise",
            "lomo",
            "schedule_free",
            "torchao",
            "torch_4bit",
            "torch_8bit",
            "flash",
        )
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


def _builds_model_optimizer(trainer):
    """True when Trainer.create_optimizer builds the optimizer (so optim = 'adalomo' gets the model). Grouped
    learning rates (embedding_learning_rate, loraplus_lr_ratio) and custom create_optimizer overrides build
    parameter groups instead."""
    args = getattr(trainer, "args", None)
    if getattr(args, "embedding_learning_rate", None) is not None:
        return False
    if getattr(args, "loraplus_lr_ratio", None) is not None:
        return False
    if getattr(trainer, "optimizer_cls_and_kwargs", None) is not None:
        return False
    method = getattr(type(trainer), "create_optimizer", None)
    module = getattr(method, "__module__", "") or ""
    return module.startswith(("transformers.", "trl.")) or getattr(method, "__qualname__", "") == (
        "UnslothTrainer.create_optimizer"
    )


_SGD_NOTE = (
    "This changes the optimizer: SGD usually needs a much larger learning_rate than AdamW. Set "
    f"{_ENV}=0 to turn this off."
)


def _decide(trainer, model, mode):
    """(tier, message): tier is None, 'in_backward', 'sgd' (LOMO's update) or 'adalomo'."""
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
    if mode == "lomo":
        return "sgd", (
            f"Unsloth: training with LOMO's update, plain SGD stepped inside backward ({_ENV}=lomo). {_SGD_NOTE}"
        )
    if mode == "adalomo":
        if not _lomo_available():
            return None, f"Unsloth: {_ENV}=adalomo needs `pip install lomo-optim`."
        if not _builds_model_optimizer(trainer):
            return None, (
                f"Unsloth: {_ENV}=adalomo ignored: this trainer builds its own parameter groups "
                "(grouped learning rates or a custom optimizer), which AdaLomo cannot take."
            )
        return "adalomo", f"Unsloth: using AdaLomo ({_ENV}=adalomo)."
    plan = plan_auto_fit(trainer, model)
    if plan["usable"] is None or plan["standard"] <= plan["usable"]:
        return None, None
    optim = _optim_name(args) or "the optimizer"
    head = (
        f"Unsloth: full finetuning of {plan['params'] / 1e9:.2f}B weights is predicted to run out of memory: "
        f"{_gb(plan['standard'])} needed for gradients ({_gb(plan['grads'])}), {optim} state "
        f"({_gb(plan['states'])}) and activations, {_gb(plan['usable'])} free."
    )
    clip = (
        f" max_grad_norm = {args.max_grad_norm} now clips each tensor by its own norm."
        if (getattr(args, "max_grad_norm", 0) or 0) > 0
        else ""
    )
    if plan["in_backward"] <= plan["usable"]:
        return "in_backward", (
            f"{head} So {optim} steps each weight inside backward and frees its gradient at once "
            f"(predicted {_gb(plan['in_backward'])}); the optimizer math is unchanged.{clip} "
            f"Set {_ENV}=0 to turn this off."
        )
    if plan["lomo"] <= plan["usable"]:
        return "sgd", (
            f"{head} Even stepping {optim} inside backward needs {_gb(plan['in_backward'])} for its state, so "
            f"training uses LOMO's update instead: plain SGD stepped inside backward, no optimizer state "
            f"(predicted {_gb(plan['lomo'])}).{clip} {_SGD_NOTE}"
        )
    return None, (
        f"{head} Neither stepping inside backward ({_gb(plan['in_backward'])}) nor LOMO's update "
        f"({_gb(plan['lomo'])}) fits; consider LoRA, a shorter max_length or freezing part of the model."
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
        previous = getattr(original, "_unsloth_previous", None)
        original = original.__wrapped__
    else:
        previous = trainer.__dict__.get("training_step")  # another instance-level wrapper, or None
    wrapper_ref = weakref.ref(wrapper)

    def training_step(*args, **kwargs):
        opt = wrapper_ref()
        if opt is None:
            return original(*args, **kwargs)
        with opt.armed():
            return original(*args, **kwargs)

    training_step._unsloth_in_backward = True
    training_step._unsloth_previous = previous
    training_step.__wrapped__ = original
    trainer.training_step = training_step


def _sgd_from(optimizer):
    """LOMO's update as torch SGD over the same parameter groups (learning rates, weight decay): with no momentum
    SGD's p -= lr * (g + wd * p) is LOMO's p * (1 - lr * wd) - lr * g."""
    groups = []
    for group in optimizer.param_groups:
        keep = {k: group[k] for k in ("lr", "weight_decay", "initial_lr") if k in group}
        lr = keep.get("lr", optimizer.defaults.get("lr", 1e-3))
        keep["lr"] = float(lr) if not torch.is_tensor(lr) else lr
        groups.append(dict(keep, params = list(group["params"])))
    return torch.optim.SGD(groups, lr = groups[0]["lr"] if groups else 1e-3, momentum = 0.0)


def _drop_previous(trainer):
    previous = trainer.__dict__.pop("_unsloth_in_backward_optimizer", None)
    if previous is not None:
        previous.close()
    trainer.__dict__.pop(_MARK, None)
    step = trainer.__dict__.get("training_step")
    if getattr(step, "_unsloth_in_backward", False):
        previous = getattr(step, "_unsloth_previous", None)
        if previous is None:
            trainer.__dict__.pop("training_step", None)
        else:
            trainer.training_step = previous


def _convert(
    trainer,
    optimizer,
    message,
    tier = "in_backward",
):
    base = _unwrap(optimizer)
    if isinstance(base, InBackwardOptimizer):
        return optimizer
    if tier == "sgd":
        base = _sgd_from(base)
    if _step_kind(base) is None:
        name = type(base).__name__
        print(f"Unsloth: not stepping {name} inside backward: it is not a per-weight optimizer.")
        return optimizer
    max_norm = float(getattr(trainer.args, "max_grad_norm", 0) or 0)
    wrapper = InBackwardOptimizer(
        base, max_grad_norm = max_norm, model = getattr(trainer, "model", None)
    )
    setattr(trainer, _MARK, True)
    setattr(trainer, _TIER, tier)
    trainer._unsloth_in_backward_optimizer = wrapper
    _wrap_clip(trainer)
    _arm_training_step(trainer, wrapper)
    if message:
        print(message)
    return wrapper


def _resume_dir(trainer):
    resume = trainer.__dict__.get("_unsloth_resume_from_checkpoint")
    if not resume:
        return None
    if isinstance(resume, bool):
        try:
            from transformers.trainer_utils import get_last_checkpoint
            return get_last_checkpoint(trainer.args.output_dir)
        except Exception:
            return None
    return str(resume)


def _recorded_tier(trainer):
    """(tier recorded in the checkpoint being resumed, resuming?). A checkpoint without the file was trained with
    the standard step."""
    folder = _resume_dir(trainer)
    if not folder:
        return None, False
    try:
        with open(os.path.join(folder, _TIER_FILE), encoding = "utf-8") as f:
            return json.load(f).get("tier") or "standard", True
    except FileNotFoundError:
        return "standard", True
    except Exception:
        return None, True


def _keep_family_on_resume(trainer, model, mode, tier, message):
    """An SGD-tier checkpoint resumes with the SGD tier, and an AdamW checkpoint never switches to it."""
    recorded, resuming = _recorded_tier(trainer)
    if not resuming or mode != "auto" or recorded is None:
        return tier, message
    if recorded == "sgd":
        if _blocker(trainer, model, False) is not None:
            return tier, message
        return "sgd", (
            "Unsloth: this checkpoint was trained with LOMO's update (plain SGD stepped inside backward), so the "
            f"resumed run keeps it. Set {_ENV}=0 to turn this off."
        )
    if tier == "sgd":
        optim = _optim_name(getattr(trainer, "args", None)) or "the optimizer"
        return None, (
            f"Unsloth: full finetuning would need LOMO's update to fit, but this checkpoint was trained with {optim}, "
            f"so the resumed run keeps {optim} ({_ENV}=lomo forces the switch)."
        )
    return tier, message


def _install_tier_record(trainer):
    try:
        from transformers import TrainerCallback
    except Exception:
        return
    ref = weakref.ref(trainer)

    class _RecordTier(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            trainer = ref()
            tier = getattr(trainer, _TIER, None) if trainer is not None else None
            root = args.output_dir
            run_dir = getattr(trainer, "_get_output_dir", None) if trainer is not None else None
            if callable(run_dir):
                try:  # hyperparameter search trials save under output_dir/run-<id>
                    root = run_dir(trial = getattr(trainer, "_trial", None))
                except Exception:
                    root = args.output_dir
            folder = os.path.join(root, f"checkpoint-{state.global_step}")
            if tier in (None, "standard") or not os.path.isdir(folder):
                return
            try:
                with open(os.path.join(folder, _TIER_FILE), "w", encoding = "utf-8") as f:
                    json.dump({"tier": tier}, f)
            except OSError:
                pass

    try:
        trainer.add_callback(_RecordTier())
    except Exception:
        pass


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
        _drop_previous(trainer)  # a new optimizer for this trainer (another train() / model_init)
        mode = auto_fit_mode()
        tier, message = (None, None) if mode == "off" else _decide(trainer, model, mode)
        tier, message = _keep_family_on_resume(trainer, model, mode, tier, message)
        setattr(trainer, _TIER, "adalomo" if tier == "adalomo" else "standard")
        if tier == "adalomo":
            _set_optim(trainer.args, tier)
            setattr(trainer, _MARK, True)
        if message and tier not in ("in_backward", "sgd"):
            print(message)
        result = original(*args, **kwargs)
        optimizer = getattr(trainer, "optimizer", None) or result
        fix_adalomo_state(optimizer)
        if tier not in ("in_backward", "sgd") or optimizer is None:
            return result
        trainer.optimizer = _convert(trainer, optimizer, message, tier)
        return trainer.optimizer

    create_optimizer.__wrapped__ = original
    trainer.create_optimizer = create_optimizer
    train = getattr(trainer, "train", None)
    if train is not None:

        def train_and_record_resume(*args, **kwargs):
            resume = kwargs.get("resume_from_checkpoint", args[0] if args else None)
            trainer.__dict__["_unsloth_resume_from_checkpoint"] = resume
            return train(*args, **kwargs)

        train_and_record_resume.__wrapped__ = train
        trainer.train = train_and_record_resume
    _install_tier_record(trainer)
    setattr(trainer, _INSTALLED, True)
    return True
