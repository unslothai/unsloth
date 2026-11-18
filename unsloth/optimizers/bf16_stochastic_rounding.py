# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bfloat16 full finetuning: step bfloat16 weights in float32, round back stochastically (UNSLOTH_BF16_STOCHASTIC_ROUNDING=0 disables).

bfloat16 keeps 8 mantissa bits, so round-to-nearest drops updates far smaller than a weight's spacing.

On CUDA with Triton, torch AdamW weights take one kernel per weight: read the bfloat16 weight, gradient and
moments, do AdamW's float32 arithmetic in registers, write all three back with stochastic rounding (Philox
seeded by (param index, step), so resumes and DDP replicas draw alike). bitsandbytes keeps its own 8-bit
state, so its weights still step a float32 copy, rounded back in one pass. UNSLOTH_BF16_SR_FUSED=0 keeps
the float32-copy route with torch.randint draws everywhere.
"""

import os
import types

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None

__all__ = [
    "enable_bf16_stochastic_rounding",
    "stochastic_round_to_bf16_",
    "bf16_stochastic_rounding_enabled",
]

_MARK = "_unsloth_bf16_stochastic_rounding"
_CHUNK = 1 << 22


def bf16_stochastic_rounding_enabled():
    value = os.environ.get("UNSLOTH_BF16_STOCHASTIC_ROUNDING", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _fused_enabled():
    """UNSLOTH_BF16_SR_FUSED=0 keeps the float32-copy path (torch.randint draws) on CUDA too."""
    value = os.environ.get("UNSLOTH_BF16_SR_FUSED", "1").strip().lower()
    return triton is not None and value not in ("0", "false", "no", "off")


if triton is not None:

    @triton.jit
    def _round_bf16(x, random):
        # Up with probability low / 2**16: E[result] = x. Same law as (bits + U[0, 2**16)) & -65536.
        bits = x.to(tl.uint32, bitcast = True)
        low = bits & 65535
        upper = bits & 4294901760
        special = (bits & 2139095040) == 2139095040
        rounded = tl.where((random & 65535) < low, upper + 65536, upper)
        return tl.where(special, bits, rounded).to(tl.float32, bitcast = True).to(tl.bfloat16)

    @triton.jit(do_not_specialize = ["N", "SEED"])
    def _round_kernel(DST, SRC, N, SEED, BLOCK: tl.constexpr):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = i < N
        x = tl.load(SRC + i, valid, 0).to(tl.float32)
        # Lane 0 of Philox(SEED, element index): what the fused AdamW kernel uses for the weight.
        r0, _, _, _ = tl.randint4x(SEED, i)
        tl.store(DST + i, _round_bf16(x, r0), valid)

    @triton.jit(do_not_specialize = ["N", "SEED"])
    def _adamw_kernel(
        P,
        G,
        M,
        V,
        N,
        SEED,
        DECAY,  # 1 - lr * weight_decay
        W1,  # 1 - beta1
        B2,
        W2,  # 1 - beta2
        NEG_STEP,  # -lr / (1 - beta1**t)
        BC2_SQRT,  # sqrt(1 - beta2**t)
        EPS,
        BLOCK: tl.constexpr,
    ):
        # torch.optim.AdamW's foreach float32 arithmetic, op for op, then one rounding per output.
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = i < N
        p = tl.load(P + i, valid, 0).to(tl.float32)
        g = tl.load(G + i, valid, 0).to(tl.float32)
        m = tl.load(M + i, valid, 0).to(tl.float32)
        v = tl.load(V + i, valid, 0).to(tl.float32)
        p = p * DECAY
        # torch.lerp: from the far end once the weight reaches 0.5, so beta1 = 0 gives exactly g
        m = tl.where(W1 < 0.5, m + W1 * (g - m), g - (g - m) * (1.0 - W1))
        v = v * B2
        v = v + W2 * g * g
        denom = tl.div_rn(tl.sqrt_rn(v), BC2_SQRT) + EPS
        p = p + NEG_STEP * tl.div_rn(m, denom)
        r0, r1, r2, _ = tl.randint4x(SEED, i)
        tl.store(P + i, _round_bf16(p, r0), valid)
        tl.store(M + i, _round_bf16(m, r1), valid)
        tl.store(V + i, _round_bf16(v, r2), valid)


_BLOCK = 1024


def _grid(n):
    return ((n + _BLOCK - 1) // _BLOCK,)


_TRITON_OK = {}


def _triton_ok(device):
    """Compile and run both kernels once per device; any failure keeps the float32-copy route there."""
    key = str(device)
    if key not in _TRITON_OK:
        try:
            with torch.cuda.device(device):
                p, g, m, v = (torch.zeros(2, dtype = torch.bfloat16, device = device) for _ in range(4))
                seed = 1 << 40  # the int64 specialisation real seeds use
                _adamw_kernel[(1,)](
                    p, g, m, v, 2, seed, 1.0, 0.1, 0.999, 0.001, -1e-3, 1.0, 1e-8,
                    BLOCK = _BLOCK, enable_fp_fusion = False,
                )  # fmt: skip
                _round_kernel[(1,)](
                    p, torch.ones(2, device = device), 2, seed, BLOCK = _BLOCK, enable_fp_fusion = False
                )
                torch.cuda.synchronize(device)
            _TRITON_OK[key] = bool(p.float().eq(1).all().item())
        except Exception as exc:
            print(
                f"Unsloth: one-pass bfloat16 stochastic rounding is unavailable on {device} ({exc})."
            )
            _TRITON_OK[key] = False
    return _TRITON_OK[key]


def _round_cuda_(dst, src, seed):
    n = src.numel()
    if n:
        with torch.cuda.device(dst.device):
            _round_kernel[_grid(n)](dst, src, n, seed, BLOCK = _BLOCK, enable_fp_fusion = False)
    return dst


def stochastic_round_to_bf16_(
    dst,
    src,
    generator = None,
):
    """dst (bfloat16) <- src (float32) rounded stochastically (unbiased: E[round(x)] = x)."""
    flat_src = src.reshape(-1)
    flat_dst = dst.view(-1)
    for start in range(0, flat_src.numel(), _CHUNK):
        bits = flat_src[start : start + _CHUNK].contiguous().view(torch.int32)
        noise = torch.randint(
            0, 1 << 16, bits.shape, dtype = torch.int32, device = bits.device, generator = generator
        )
        rounded = (bits + noise) & -65536
        special = (bits & 0x7F800000) == 0x7F800000
        rounded = torch.where(special, bits, rounded)
        flat_dst[start : start + _CHUNK].copy_(rounded.view(torch.float32))
    return dst


def _is_dtensor(p):
    return type(p).__name__ == "DTensor" or hasattr(p, "_local_tensor")


def _tracked(p):
    return (
        p.dtype == torch.bfloat16
        and p.requires_grad
        and not _is_dtensor(p)
        and p.layout == torch.strided
        and p.is_contiguous()
        and p.device.type in ("cuda", "cpu", "xpu")
    )


def _generator(optimizer, device, seed):
    generators = optimizer.__dict__.setdefault("_unsloth_sr_generators", {})
    key = str(device)
    if key not in generators:
        g = torch.Generator(device = device)
        # The same seed on every rank: DDP replicas round identically and stay in sync.
        g.manual_seed(int(seed) + 0x5EED)
        generators[key] = g
    return generators[key]


def _index(optimizer, p):
    index = optimizer.__dict__.setdefault("_unsloth_sr_index", {})
    return index.setdefault(id(p), len(index))


def _seed_for(optimizer, p, seed, step):
    """Seeded by (param index, optimizer step): a resumed run draws what an uninterrupted one would."""
    return (int(seed) + 0x5EED + _index(optimizer, p) * 1000003 + int(step) * 2654435761) & (
        0x7FFFFFFFFFFFFFFF
    )


def _step_of(optimizer, p):
    step = (optimizer.state.get(p) or {}).get("step", 0)
    return int(step.item()) if torch.is_tensor(step) else int(step)


def _reseed(optimizer, p, seed):
    g = _generator(optimizer, p.device, seed)
    g.manual_seed(_seed_for(optimizer, p, seed, _step_of(optimizer, p)))
    return g


class _Float32Step:
    """Step a float32 copy of one bfloat16 weight, then round it back into the weight in place.

    The weight's .data is never swapped: changing its dtype resets the weight's AccumulateGrad node, and
    DDP's allreduce hooks live on that node, so DDP would silently stop syncing that gradient from the
    next backward. The optimizer steps the copy under the weight's own state (moved to the copy's key for
    the call), and only the copy is float32."""

    def __init__(
        self,
        optimizer,
        p,
        seed,
        triton_round = False,
    ):
        self.optimizer, self.p, self.seed = optimizer, p, seed
        self.triton_round = (
            triton_round and p.device.type == "cuda" and _fused_enabled() and _triton_ok(p.device)
        )

    def __enter__(self):
        p = self.p
        copy = torch.nn.Parameter(p.detach().float())
        copy.grad = p.grad.float()
        state = self.optimizer.state
        if p in state:
            state[copy] = state.pop(p)
        self.copy = copy
        return copy

    def __exit__(self, *exc):
        p, copy = self.p, self.copy
        self.copy = None
        state = self.optimizer.state
        if copy in state:
            state[p] = state.pop(copy)
        if exc[0] is None:
            stepped = copy.detach()
            if self.triton_round:
                try:
                    seed = _seed_for(self.optimizer, p, self.seed, _step_of(self.optimizer, p))
                    _round_cuda_(p.detach(), stepped, seed)
                    _count(self.optimizer, "_unsloth_sr_triton_rounds")
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as error:  # a launch failing after the device probe passed
                    _TRITON_OK[str(p.device)] = False
                    print(
                        f"Unsloth: one-pass bfloat16 stochastic rounding failed on {p.device} ({error}); "
                        "using the float32 route there."
                    )
                    self.triton_round = False
            if not self.triton_round:
                stochastic_round_to_bf16_(
                    p.detach(), stepped, _reseed(self.optimizer, p, self.seed)
                )
        return False


_MOMENT_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


def _moments(state, p, dtype):
    # By name: a scalar parameter's fp32 `step` has its shape too.
    return [
        key
        for key in _MOMENT_KEYS
        if torch.is_tensor(state.get(key))
        and state[key].dtype == dtype
        and state[key].shape == p.shape
    ]


def _upcast_state(state, p):
    for key in _moments(state, p, torch.bfloat16):
        state[key] = state[key].float()


def _store_state(optimizer, state, p, seed):
    for key in _moments(state, p, torch.float32):
        value = state[key]
        stored = torch.empty_like(value, dtype = torch.bfloat16)
        stochastic_round_to_bf16_(stored, value, _generator(optimizer, value.device, seed))
        state[key] = stored


def _wrap_bitsandbytes(optimizer, tracked, seed):
    original = optimizer.update_step

    def update_step(self, group, p, gindex, pindex):
        if id(p) not in tracked or p.grad is None or p.dtype != torch.bfloat16:
            return original(group, p, gindex, pindex)
        # bitsandbytes keeps its own 8-bit state and needs p and its gradient in one dtype: step a float32
        # copy under p's state, round it back into p in one pass.
        with _Float32Step(self, p, seed, triton_round = True) as copy:
            return original(group, copy, gindex, pindex)

    optimizer.update_step = types.MethodType(update_step, optimizer)


def _step_hooks(optimizer):
    return bool(
        getattr(optimizer, "_optimizer_step_pre_hooks", None)
        or getattr(optimizer, "_optimizer_step_post_hooks", None)
    )


def _wrap_torch(optimizer, tracked, seed):
    original = optimizer.step
    # The class step without torch's hook dispatch: hooks run once around `update`, with the real weights.
    cls_step = type(optimizer).step
    unhooked = (
        getattr(cls_step, "__wrapped__", None) if getattr(cls_step, "hooked", False) else None
    )
    if unhooked is None or getattr(original, "__func__", None) is not cls_step:
        unhooked = None

    def update(self, closure = None):
        """Step self.param_groups' single group of tracked weights inside torch's step hook dispatch (so state
        offload, gradient and DP hooks see the real weights): the one-pass kernel where it applies, else a float32
        copy per weight, rounded back in place."""
        if closure is not None:  # e.g. state offload's restore, swapped in by its pre hook
            with torch.enable_grad():
                closure()
        (group,) = self.param_groups
        params = [p for p in group["params"] if p.grad is not None]  # pre hooks may drop gradients
        if params and _fused_group(self, group):
            fused = [p for p in params if _fused_param(self, p)]
            left = {id(p) for p in _fused_adamw_step(self, group, fused, seed)} if fused else set()
            done = {id(p) for p in fused if id(p) not in left}
            params = [p for p in params if id(p) not in done]
        try:
            for p in params:
                state = self.state.get(p)
                if state:
                    _upcast_state(state, p)
                with _Float32Step(self, p, seed) as copy:
                    self.param_groups = [dict(group, params = [copy])]
                    if unhooked is not None:
                        unhooked(self)
                    else:
                        original()
                _store_state(self, self.state[p], p, seed)
        finally:
            self.param_groups = [group]

    profile_hook_step = getattr(torch.optim.Optimizer, "profile_hook_step", None)
    hooked = profile_hook_step(update) if profile_hook_step is not None else update

    # A bound method: torch's LR scheduler wraps `optimizer.step.__func__`.
    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        groups = self.param_groups
        try:
            for group in groups:
                rest = [p for p in group["params"] if id(p) not in tracked]
                mine = [p for p in group["params"] if id(p) in tracked and p.grad is not None]
                if rest:
                    self.param_groups = [dict(group, params = rest)]
                    original()
                for p in mine:  # indices in parameter order, whichever route steps them
                    _index(self, p)
                # Step hooks fire per weight when the optimizer has its own (state offload, idle experts,
                # DP sync), as torch's per-weight steps did before; else once for the group.
                batches = [[p] for p in mine] if _step_hooks(self) else ([mine] if mine else [])
                for batch in batches:
                    self.param_groups = [dict(group, params = batch)]
                    hooked(self, None)
        finally:
            self.param_groups = groups
        return loss

    optimizer.step = types.MethodType(step, optimizer)


def _count(
    optimizer,
    name,
    n = 1,
):
    optimizer.__dict__[name] = optimizer.__dict__.get(name, 0) + n


def _float(x):
    if torch.is_tensor(x):
        if x.device.type != "cpu" or x.numel() != 1:
            return None
        x = x.item()
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fused_group(optimizer, group):
    """Groups the one-pass kernel reproduces: AdamW (or Adam without L2 decay), float hyperparameters."""
    if not _fused_enabled():
        return False
    if group.get("amsgrad") or group.get("maximize") or group.get("differentiable"):
        return False
    if group.get("capturable"):
        return False
    # GradScaler hands a fused optimizer its scale and overflow flag instead of unscaling: torch's kernel uses them.
    if (
        getattr(optimizer, "grad_scale", None) is not None
        or getattr(optimizer, "found_inf", None) is not None
    ):
        return False
    betas = group.get("betas", (0.9, 0.999))
    values = [
        _float(group.get("lr")),
        _float(group.get("eps")),
        _float(group.get("weight_decay", 0)),
    ]
    values += [_float(b) for b in betas]
    if len(betas) != 2 or any(v is None for v in values):
        return False
    decoupled = group.get("decoupled_weight_decay", None)
    if decoupled is None:  # torch < 2.7: AdamW groups carry no flag, the class decides
        decoupled = isinstance(optimizer, torch.optim.AdamW)
    return bool(decoupled) or values[2] == 0


def _fused_param(optimizer, p):
    g = p.grad
    if (
        p.device.type != "cuda"
        or g is None
        or g.is_sparse
        or g.layout != torch.strided
        or g.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or g.device != p.device
        or g.shape != p.shape
        or not g.is_contiguous()
        or not _triton_ok(p.device)
    ):
        return False
    state = optimizer.state.get(p)
    if not state:
        return True
    for key in ("exp_avg", "exp_avg_sq"):
        value = state.get(key)
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.bfloat16
            or value.shape != p.shape
            or value.device != p.device
            or not value.is_contiguous()
        ):
            return False  # e.g. float32 moments from a checkpoint: the float32 route converts them
    return torch.is_tensor(state.get("step")) and state["step"].numel() == 1


def _scalar_dtype():
    try:
        from torch.optim.optimizer import _get_scalar_dtype
        return _get_scalar_dtype()
    except Exception:
        return torch.float32


def _fused_adamw_step(optimizer, group, params, seed):
    """One kernel per weight: AdamW in float32 registers, weight and both bfloat16 moments rounded stochastically.
    Returns the weights it could not step (a failed launch), for the float32 route."""
    lr, eps, weight_decay = (_float(group[k]) for k in ("lr", "eps", "weight_decay"))
    beta1, beta2 = (_float(b) for b in group["betas"])
    on_device = bool(group.get("fused") or group.get("capturable"))
    states = []
    for p in params:
        state = optimizer.state[p]
        if not state:  # what torch.optim.AdamW creates on a first step
            state["step"] = (
                torch.zeros((), dtype = _scalar_dtype(), device = p.device)
                if on_device
                else torch.tensor(0.0, dtype = _scalar_dtype())
            )
            state["exp_avg"] = torch.zeros_like(p, memory_format = torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, memory_format = torch.preserve_format)
        states.append(state)
    steps = [state["step"] for state in states]
    cpu = [s for s in steps if s.device.type == "cpu"]
    dev = [s for s in steps if s.device.type != "cpu"]
    if cpu:
        torch._foreach_add_(cpu, 1)
    if dev:
        torch._foreach_add_(dev, 1)
    # One host read per device for its step counters (torch's fused AdamW keeps them on the GPU).
    values = {}
    for device in {s.device for s in dev}:
        here = [s for s in dev if s.device == device]
        values.update(zip(map(id, here), torch.stack([s.float() for s in here]).tolist()))
    decay = 1 - lr * weight_decay
    stepped, left = [], []
    for i, (p, state, step) in enumerate(zip(params, states, steps)):
        t = step.item() if step.device.type == "cpu" else values[id(step)]
        bc1 = 1 - beta1**t
        bc2 = 1 - beta2**t
        n = p.numel()
        if n:
            try:
                with torch.cuda.device(p.device):
                    _adamw_kernel[_grid(n)](
                        p.data,
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        n,
                        _seed_for(optimizer, p, seed, int(t)),
                        decay,
                        1 - beta1,
                        beta2,
                        1 - beta2,
                        (lr / bc1) * -1,
                        bc2**0.5,
                        eps,
                        BLOCK = _BLOCK,
                        enable_fp_fusion = False,
                    )
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:
                # A launch failed after the device probe passed (a specialisation the probe did not compile).
                # Nothing was written for this weight or the ones after it: undo their step count and hand them
                # to the float32 route, which every later step on this device takes too.
                _TRITON_OK[str(p.device)] = False
                print(
                    f"Unsloth: one-pass bfloat16 stochastic rounding failed on {p.device} ({exc}); "
                    "using the float32 route there."
                )
                for s in steps[i:]:
                    s.sub_(1)
                left = params[i:]
                break
        stepped.append(p)
    increment = getattr(torch.autograd.graph, "increment_version", None)
    if increment is not None and stepped:
        increment([p.data for p in stepped])
    _count(optimizer, "_unsloth_sr_fused_steps", len(stepped))
    return left


def _kind(optimizer):
    try:
        from bitsandbytes.optim.optimizer import Optimizer8bit
        if isinstance(optimizer, Optimizer8bit) and hasattr(optimizer, "update_step"):
            return "bitsandbytes"
    except Exception:
        pass
    if type(optimizer) in (torch.optim.AdamW, torch.optim.Adam):
        return "torch"
    return None


def enable_bf16_stochastic_rounding(
    optimizer,
    seed = 3407,
    verbose = True,
):
    """Step this optimizer's bfloat16 weights in float32 with stochastic rounding back to bfloat16.
    Returns the number of weights it covers (0: nothing to do, unsupported optimizer, or turned off).
    Call before an LR scheduler is built over the optimizer. Idempotent."""
    if optimizer is None or not bf16_stochastic_rounding_enabled():
        return 0
    inner = getattr(optimizer, "optimizer", None)
    if inner is not None and hasattr(inner, "param_groups"):  # accelerate's AcceleratedOptimizer
        optimizer = inner
    if getattr(optimizer, _MARK, None) is not None:
        return getattr(optimizer, _MARK)
    kind = _kind(optimizer)
    if kind is None:
        return 0
    params = [p for group in optimizer.param_groups for p in group["params"] if _tracked(p)]
    if not params:
        return 0
    tracked = {id(p) for p in params}
    if kind == "bitsandbytes":
        _wrap_bitsandbytes(optimizer, tracked, seed)
    else:
        _wrap_torch(optimizer, tracked, seed)
    setattr(optimizer, _MARK, len(params))
    if verbose:
        print(
            f"Unsloth: bfloat16 full finetuning steps {len(params)} bfloat16 weights in float32 and "
            "rounds them back stochastically, so updates smaller than bfloat16's spacing are not lost. "
            "Set UNSLOTH_BF16_STOCHASTIC_ROUNDING=0 to turn this off."
        )
    return len(params)


def _full_finetuning(model):
    seen = set()
    while model is not None and id(model) not in seen:
        seen.add(id(model))
        flag = getattr(model, "_unsloth_full_finetuning", None)
        if flag is not None:
            return bool(flag)
        model = getattr(model, "module", None)
    return False


def patch_trainer_bf16_stochastic_rounding(Trainer):
    """create_scheduler: the optimizer exists and the LR scheduler has not yet wrapped its step."""
    original = Trainer.create_scheduler
    if getattr(original, _MARK, False):
        return

    def create_scheduler(
        self,
        num_training_steps,
        optimizer = None,
        *args,
        **kwargs,
    ):
        target = optimizer if optimizer is not None else getattr(self, "optimizer", None)
        if getattr(self, "lr_scheduler", None) is None and _full_finetuning(
            getattr(self, "model", None)
        ):
            try:
                enable_bf16_stochastic_rounding(target, seed = getattr(self.args, "seed", 3407))
            except Exception as exc:  # never block training on this
                print(f"Unsloth: bfloat16 stochastic rounding was not enabled ({exc}).")
        return original(self, num_training_steps, optimizer, *args, **kwargs)

    setattr(create_scheduler, _MARK, True)
    create_scheduler.__wrapped__ = original
    create_scheduler.__doc__ = original.__doc__
    Trainer.create_scheduler = create_scheduler
