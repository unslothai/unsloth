# SPDX-License-Identifier: AGPL-3.0-or-later

import math
import torch
from torch import Tensor
from typing import List

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:
    _BATCHED_ARGUMENTS = tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 3)

    @triton.jit(do_not_specialize = ["N"])
    def _adamw_tile(
        P,
        G,
        M,
        V,
        STEP,
        LR,
        N,
        SEED,
        OFFSET,
        TILE,
        B1: tl.constexpr,  # beta1
        B2: tl.constexpr,  # beta2
        EPS: tl.constexpr,  # Adam eps
        WD: tl.constexpr,  # weight decay
        BLOCK: tl.constexpr,  # elements per program
    ):
        i = TILE.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = i < N
        p = tl.load(P + i, valid, 0).to(tl.float32)
        g = tl.load(G + i, valid, 0).to(tl.float32)
        m = tl.load(M + i, valid, 0)
        v = tl.load(V + i, valid, 0)
        step = tl.load(STEP)
        lr = tl.load(LR)
        m = m + (g - m) * (1.0 - B1)
        v = v + (g * g - v) * (1.0 - B2)
        bc1 = 1.0 - tl.exp(tl.log(B1) * step.to(tl.float32))
        bc2 = 1.0 - tl.exp(tl.log(B2) * step.to(tl.float32))
        p = p * (1.0 - lr * WD) - lr * (m / bc1) / (tl.sqrt(v / bc2) + EPS)
        bits = p.to(tl.uint32, bitcast = True)
        low = bits & 65535
        upper = bits & 4294901760
        random = (
            tl.randint(tl.load(SEED).to(tl.uint64) + step.to(tl.uint64), (i + OFFSET).to(tl.uint32))
            & 65535
        )
        special = (bits & 2139095040) == 2139095040
        rounded = tl.where((random < low) & ~special, upper + 65536, upper)
        # Preserve exceptional values instead of rounding NaN payloads into infinity.
        result = tl.where(special, bits, rounded).to(tl.float32, bitcast = True)
        tl.store(P + i, result.to(tl.bfloat16), valid)
        tl.store(M + i, m, valid)
        tl.store(V + i, v, valid)

    @triton.jit(do_not_specialize = ["sizes", "offsets"])
    def _adamw_batched(
        P,
        G,
        M,
        V,
        STEP,
        SEED,
        LR,
        sizes,
        offsets,
        B1: tl.constexpr,  # beta1
        B2: tl.constexpr,  # beta2
        EPS: tl.constexpr,  # Adam eps
        WD: tl.constexpr,  # weight decay
        BLOCK: tl.constexpr,  # elements per program
    ):
        pid = tl.program_id(0).to(tl.int64)
        start = tl.full((), 0, tl.int64)
        for j in tl.static_range(len(sizes)):
            n = tl.cast(sizes[j], tl.int64)
            offset = tl.cast(offsets[j], tl.int64)
            stop = start + (n + BLOCK - 1) // BLOCK
            if (pid >= start) & (pid < stop):
                _adamw_tile(
                    P[j],
                    G[j],
                    M[j],
                    V[j],
                    STEP[j],
                    LR,
                    n,
                    SEED[j],
                    offset,
                    pid - start,
                    B1,
                    B2,
                    EPS,
                    WD,
                    BLOCK,
                )
            start = stop

    @triton.jit(do_not_specialize = ["N", "OFFSET"])
    def _adamw_single(
        P,
        G,
        M,
        V,
        STEP,
        LR,
        N,
        SEED,
        OFFSET,
        B1: tl.constexpr,  # beta1
        B2: tl.constexpr,  # beta2
        EPS: tl.constexpr,  # Adam eps
        WD: tl.constexpr,  # weight decay
        BLOCK: tl.constexpr,  # elements per program
    ):
        _adamw_tile(
            P,
            G,
            M,
            V,
            STEP,
            LR,
            N,
            SEED,
            OFFSET.to(tl.int64),
            tl.program_id(0),
            B1,
            B2,
            EPS,
            WD,
            BLOCK,
        )

    def _launch(
        p: List[Tensor],
        g: List[Tensor],
        m: List[Tensor],
        v: List[Tensor],
        step: List[Tensor],
        lr: Tensor,
        seed: List[Tensor],
        offsets: List[int],
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        sizes = tuple(x.numel() for x in p)
        tiles = sum((n + 255) // 256 for n in sizes)
        if not tiles:
            return
        with torch.cuda.device(p[0].device):
            if not _BATCHED_ARGUMENTS:
                for pi, gi, mi, vi, si, seedi, oi, n in zip(p, g, m, v, step, seed, offsets, sizes):
                    if n:
                        _adamw_single[((n + 255) // 256,)](
                            pi,
                            gi,
                            mi,
                            vi,
                            si,
                            lr,
                            n,
                            seedi,
                            oi,
                            beta1,
                            beta2,
                            eps,
                            weight_decay,
                            256,
                            enable_fp_fusion = False,
                        )
                return
            _adamw_batched[(tiles,)](
                tuple(p),
                tuple(g),
                tuple(m),
                tuple(v),
                tuple(step),
                tuple(seed),
                lr,
                sizes,
                tuple(offsets),
                beta1,
                beta2,
                eps,
                weight_decay,
                256,
                enable_fp_fusion = False,
            )

    # Keep this fused update opaque so Inductor cannot change its rounding.
    _cuda_update = torch.library.custom_op(
        "unsloth::adamw_bf16_update", mutates_args = {"p", "m", "v"}
    )(_launch)

    @_cuda_update.register_fake
    def _fake_update(p, g, m, v, step, lr, seed, offsets, beta1, beta2, eps, weight_decay):
        return None


def _dtensor_class():
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        try:
            from torch.distributed._tensor import DTensor
        except ImportError:
            return None
    return DTensor


def _is_dtensor(t):
    cls = _dtensor_class()
    return cls is not None and isinstance(t, cls)


def _local(t):
    return t._local_tensor if _is_dtensor(t) else t


def _plain_parameter(p):
    return type(p) in (Tensor, torch.nn.Parameter) and p.layout == torch.strided


def _dense_parameter(p):
    """Plain strided tensor, or a DTensor whose local shard is one contiguous run (dim-0 sharding)."""
    if not _is_dtensor(p):
        return _plain_parameter(p)
    local = p._local_tensor
    if not _plain_parameter(local) or local.layout != torch.strided:
        return False
    return _shard_offset(p) is not None


_SHARD_OFFSETS = None


def _shard_offset(p):
    """Flat index of the local shard's first element in the whole tensor, None if not contiguous."""
    global _SHARD_OFFSETS
    if not _is_dtensor(p):
        return 0
    if _SHARD_OFFSETS is None:
        from torch.utils.weak import WeakIdKeyDictionary
        _SHARD_OFFSETS = WeakIdKeyDictionary()
    key = (tuple(p.shape), tuple(p.placements), id(p.device_mesh), tuple(p._local_tensor.shape))
    cached = _SHARD_OFFSETS.get(p)
    if cached is not None and cached[0] == key:
        return cached[1]
    offset = _compute_shard_offset(p)
    _SHARD_OFFSETS[p] = (key, offset)
    return offset


def _compute_shard_offset(p):
    try:
        from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    except ImportError:
        from torch.distributed._tensor._utils import compute_local_shape_and_global_offset
    shape = tuple(p.shape)
    local_shape, offset = compute_local_shape_and_global_offset(shape, p.device_mesh, p.placements)
    if tuple(local_shape) != tuple(p._local_tensor.shape):
        return None
    if len(shape) == 0:
        return 0
    if tuple(local_shape[1:]) != shape[1:] or any(offset[1:]):
        return None
    return int(offset[0]) * math.prod(shape[1:])


def prepare_bf16_training(model):
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        if (
            not _plain_parameter(p)
            or p.dtype not in (torch.float32, torch.bfloat16)
            or not p.is_contiguous()
            or p.device.type not in ("cpu", "cuda")
        ):
            raise ValueError(
                "BF16 training requires contiguous plain FP32/BF16 CPU or CUDA parameters "
                "(under FSDP2, call prepare_bf16_training before the model is sharded)"
            )
        if p.grad is not None:
            raise ValueError("Prepare BF16 training before creating gradients")
    with torch.no_grad():
        for p in params:
            p.data = p.data.to(torch.bfloat16)
    return model


def _cpu_update(p, g, m, v, step, lr, seed, offset, beta1, beta2, eps, weight_decay):
    generator = torch.Generator(device = "cpu")
    # CPU generators keep only the low 32 seed bits; sr_seed holds the index in the high 32 (splitmix64).
    z = (int(seed) + int(step)) % (2**63 - 1)
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & (2**64 - 1)
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & (2**64 - 1)
    generator.manual_seed((z ^ (z >> 31)) & 0xFFFFFFFF)
    gf = g.float()
    # Not lerp_: MSVC CPU lerp rounds tail elements differently, so shards would miss unsharded bits.
    m.mul_(beta1).add_(gf * (1 - beta1))
    v.mul_(beta2).add_(gf.square() * (1 - beta2))
    t = int(step)
    value = p.float() * (1 - lr * weight_decay)
    value = value - lr * (m / (1 - beta1**t)) / ((v / (1 - beta2**t)).sqrt() + eps)
    bits = value.contiguous().view(torch.int32)
    # Draw the whole stream up to this shard so each element matches its unsharded draw.
    random = torch.randint(
        65536, (int(offset) + bits.numel(),), generator = generator, dtype = torch.int32
    )
    random = random[int(offset) :].view(bits.shape)
    upper = bits & -65536
    rounded = upper + ((random < (bits & 65535)).to(torch.int32) << 16)
    special = (bits & 2139095040) == 2139095040
    p.copy_(torch.where(special, bits, rounded).view(torch.float32))


def _moment_like(value, p):
    if not _is_dtensor(p):
        if _is_dtensor(value):
            value = value.full_tensor()
        return value.to(device = p.device, dtype = torch.float32).contiguous()
    DTensor = _dtensor_class()
    if _is_dtensor(value):
        if value.device_mesh != p.device_mesh or tuple(value.placements) != tuple(p.placements):
            value = value.redistribute(p.device_mesh, p.placements)
        local = value._local_tensor
    else:
        local = value.reshape(-1)[_shard_offset(p) : _shard_offset(p) + _local(p).numel()]
        local = local.view(_local(p).shape)
    local = local.to(device = _local(p).device, dtype = torch.float32).contiguous()
    if local.shape != _local(p).shape:
        raise ValueError(
            f"AdamWBF16 state shard has shape {tuple(local.shape)}, but its parameter shard has "
            f"shape {tuple(_local(p).shape)}"
        )
    return DTensor.from_local(
        local, p.device_mesh, p.placements, run_check = False, shape = p.shape, stride = p.stride()
    )


class AdamWBF16(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-3,
        betas = (0.9, 0.999),
        eps = 1e-8,
        weight_decay = 0.01,
        *,
        seed = 3407,
    ):
        if any(not math.isfinite(float(x)) or float(x) < 0 for x in (lr, eps, weight_decay)):
            raise ValueError("Invalid learning rate, epsilon or weight decay")
        if not all(0 <= b < 1 for b in betas):
            raise ValueError("Invalid Adam betas")
        super().__init__(
            params, dict(lr = lr, betas = betas, eps = eps, weight_decay = weight_decay, seed = int(seed))
        )
        self._initialize()
        self._ready = True

    def add_param_group(self, param_group):
        param_group = dict(param_group)
        values = param_group["params"]
        values = [values] if isinstance(values, Tensor) else list(values)
        param_group["params"] = values
        for p in values:
            if isinstance(p, Tensor) and not p.requires_grad and p.grad is None:
                continue
            if (
                not isinstance(p, Tensor)
                or not _dense_parameter(p)
                or p.dtype != torch.bfloat16
                or not _local(p).is_contiguous()
                or p.device.type not in ("cpu", "cuda")
                or p.numel() > 2**32
            ):
                raise ValueError(
                    "AdamWBF16 requires dense contiguous BF16 parameters with <=2**32 elements "
                    "(plain tensors or dim-0 sharded DTensors)"
                )
            if p.device.type == "cuda" and triton is None:
                raise RuntimeError("CUDA AdamWBF16 requires Triton")
        group = {**self.defaults, **param_group}
        if any(
            not math.isfinite(float(group[k])) or float(group[k]) < 0
            for k in ("lr", "eps", "weight_decay")
        ):
            raise ValueError("Invalid AdamW hyperparameters")
        if len(group["betas"]) != 2 or not all(0 <= b < 1 for b in group["betas"]):
            raise ValueError("Invalid Adam betas")
        super().add_param_group(param_group)
        if hasattr(self, "_ready"):
            self._initialize()

    def _initialize(self):
        params = [
            p
            for group in self.param_groups
            for p in group["params"]
            if p.requires_grad or p.grad is not None
        ]
        for p in params:
            if (
                not _dense_parameter(p)
                or p.dtype != torch.bfloat16
                or not _local(p).is_contiguous()
            ):
                raise ValueError("AdamWBF16 requires contiguous plain BF16 parameters")
            if p.device.type not in ("cpu", "cuda"):
                raise ValueError("AdamWBF16 supports CPU and CUDA only")
            if p.device.type == "cuda" and triton is None:
                raise RuntimeError("CUDA AdamWBF16 requires Triton")
        index = 0
        for group in self.param_groups:
            device = torch.device("cpu")
            for p in group["params"]:
                if p.requires_grad or p.grad is not None:
                    device = p.device
                    break
            group["lr"] = torch.as_tensor(group["lr"], dtype = torch.float32, device = device)
            for p in group["params"]:
                if p.grad is None:
                    index += 1
                    continue
                state = self.state[p]
                if not state:
                    state.update(
                        step = torch.zeros((), device = p.device, dtype = torch.int64),
                        exp_avg = torch.zeros_like(p, dtype = torch.float32),
                        exp_avg_sq = torch.zeros_like(p, dtype = torch.float32),
                        sr_seed = torch.tensor(
                            (int(group["seed"]) + (index << 32)) % (2**63 - 1),
                            device = p.device,
                            dtype = torch.int64,
                        ),
                    )
                index += 1

    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if not p.requires_grad and g is None:
                    continue
                if (
                    not _dense_parameter(p)
                    or p.dtype != torch.bfloat16
                    or not _local(p).is_contiguous()
                ):
                    raise ValueError("AdamWBF16 requires contiguous plain BF16 parameters")
                if g is not None and (
                    _is_dtensor(g) != _is_dtensor(p)
                    or not _dense_parameter(g)
                    or g.dtype != torch.bfloat16
                    or not _local(g).is_contiguous()
                    or _local(g).shape != _local(p).shape
                    or (_is_dtensor(p) and tuple(g.placements) != tuple(p.placements))
                ):
                    raise ValueError("AdamWBF16 requires dense contiguous BF16 gradients")
        if any(
            p.grad is not None and not self.state.get(p)
            for group in self.param_groups
            for p in group["params"]
        ):
            self._initialize()
        for group in self.param_groups:
            by_device = {}
            for p in group["params"]:
                if p.grad is not None:
                    by_device.setdefault(p.device, []).append(p)
            for device, params in by_device.items():
                lr = torch.as_tensor(group["lr"], dtype = torch.float32, device = device)
                torch._foreach_add_([self.state[p]["step"] for p in params], 1)
                if device.type == "cpu":
                    for p in params:
                        state = self.state[p]
                        _cpu_update(
                            _local(p),
                            _local(p.grad),
                            _local(state["exp_avg"]),
                            _local(state["exp_avg_sq"]),
                            state["step"],
                            lr,
                            state["sr_seed"],
                            _shard_offset(p),
                            *group["betas"],
                            group["eps"],
                            group["weight_decay"],
                        )
                    continue
                small, large = [], []
                for p in params:
                    (small if _local(p).numel() <= 65536 else large).append(p)
                for width, bucket in ((32, small), (8, large)):
                    for start in range(0, len(bucket), width):
                        batch = bucket[start : start + width]
                        states = [self.state[p] for p in batch]
                        update = _cuda_update if torch.compiler.is_compiling() else _launch
                        update(
                            [_local(p) for p in batch],
                            [_local(p.grad) for p in batch],
                            [_local(state["exp_avg"]) for state in states],
                            [_local(state["exp_avg_sq"]) for state in states],
                            [state["step"] for state in states],
                            lr,
                            [state["sr_seed"] for state in states],
                            [_shard_offset(p) for p in batch],
                            *group["betas"],
                            group["eps"],
                            group["weight_decay"],
                        )
                        if not torch.compiler.is_compiling():
                            torch.autograd.graph.increment_version([_local(p) for p in batch])
                            dtensors = [p for p in batch if _is_dtensor(p)]
                            if dtensors:
                                torch.autograd.graph.increment_version(dtensors)
        return loss

    def load_state_dict(self, state_dict):
        saved = state_dict
        super().load_state_dict(state_dict)
        for group, original_group in zip(self.param_groups, saved["param_groups"]):
            for p, old_id in zip(group["params"], original_group["params"]):
                original = saved["state"].get(old_id)
                if original is None:
                    continue
                state = self.state[p]
                # The CUDA kernel indexes moments by the parameter's element count and reads step / seed as scalars,
                # so a mismatched checkpoint must fail here, not read and write out of bounds on the next step.
                for name in ("exp_avg", "exp_avg_sq"):
                    if tuple(original[name].shape) != tuple(p.shape):
                        raise ValueError(
                            f"AdamWBF16 state {name} has shape {tuple(original[name].shape)}, "
                            f"but its parameter has shape {tuple(p.shape)}"
                        )
                for name in ("step", "sr_seed"):
                    if original[name].numel() != 1:
                        raise ValueError(f"AdamWBF16 state {name} must be a single value")
                for name in ("exp_avg", "exp_avg_sq"):
                    state[name] = _moment_like(original[name], p)
                state["step"] = original["step"].to(device = p.device, dtype = torch.int64)
                state["sr_seed"] = original["sr_seed"].to(device = p.device, dtype = torch.int64)
        self._initialize()
