# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Expert parallel over per-expert `nn.ModuleList` experts (transformers 4.x MoE layout).

Each rank keeps experts [lo, hi) under global indices. The block forward runs twice: pass 1 records expert calls,
rows are exchanged to owners and back, pass 2 replays outputs in call order. RNG is restored between passes, and
every rank enters both exchanges and their backward even when it routes or receives nothing."""

import collections
import contextlib
import copy
import functools
import os
import itertools
import operator

import torch
import torch.nn as nn

LIST = "list"  # first item of the _EP tag on a per-expert parameter: (LIST, experts, ep)
_SHARD = "_unsloth_ep_expert_list"
_UNFUSED = "unfused per-expert modules (transformers 4.x layout)"


def _ep():
    from . import ep_native
    return ep_native


def _tpn():
    from . import tp_native
    return tp_native


def is_listed(block):
    return isinstance(block.__dict__.get(_SHARD), _ListShard)


def list_shard_problem(block, ep = None):
    """Why expert parallel cannot shard `block`'s per-expert ModuleList by index, None when it can."""
    experts = block._modules.get("experts")
    if type(experts) is not nn.ModuleList:
        return "no per-expert ModuleList experts"
    E = len(experts)
    if E < 2:
        return f"{E} experts"
    if isinstance(ep, int) and ep > E:
        return f"{E} experts over {ep} ranks"
    kinds = {type(m) for m in experts}
    if len(kinds) != 1 or not any(True for _ in experts[0].parameters()):
        return "experts of different types, or without parameters"
    zoo = _zoo_fast_path(block, experts)
    if zoo is not None:
        return zoo
    # Logical shapes: a bitsandbytes 4-bit weight's packed storage can differ between experts of one checkpoint.
    shapes = {tuple((n, _logical_shape(p)) for n, p in m.named_parameters()) for m in experts}
    if len(shapes) != 1:
        return "experts with different parameter shapes"
    if any(type(b) is not torch.Tensor and b is not None for m in experts for b in m.buffers()):
        return "expert buffers"
    if getattr(block, "ep_size", 1) not in (1, None):
        return f"the block runs its own expert parallel (ep_size = {block.ep_size})"
    return None


# unsloth_zoo's speedups for per-expert lists, which expert parallel cannot shard by expert index.
_ZOO_STACKED_LORA, _ZOO_GROUPED = "UNSLOTH_MOE_STACKED_LORA", "UNSLOTH_MOE_GROUPED"


def keep_zoo_expert_loops():
    """Under expert parallel the zoo keeps per-expert LoRA and the loop (an explicit setting wins)."""
    os.environ.setdefault(_ZOO_STACKED_LORA, "0")
    os.environ.setdefault(_ZOO_GROUPED, "0")


def _zoo_fast_path(block, experts):
    if any(
        "_unsloth_stack_owner" in m.__dict__ or "weight_stack" in m._parameters
        for m in experts.modules()
    ):
        return (
            "unsloth_zoo's stacked expert LoRA (one weight_stack Parameter per projection, held by expert 0); "
            f"set {_ZOO_STACKED_LORA}=0 before get_peft_model"
        )
    if "_orig_moe_forward" in block.__dict__:
        return (
            "unsloth_zoo's grouped MoE forward (it computes every expert at once, without calling them); "
            f"set {_ZOO_GROUPED}=0 before from_pretrained"
        )
    return None


_LINEAR_LISTS = "per-expert Linear lists"


def fused_list_problem(block, ep = None):
    """Why `block`'s fused-call per-expert Linear lists (zoo GptOssExpertsBnb4bit) cannot shard, None when they can."""
    tpn = _tpn()
    chain = tpn._experts_chain(block)
    if len(chain) != 1:
        return "a wrapper around the experts module"
    problem = tpn.expert_shard_problem(block, ep)
    if not (isinstance(problem, str) and problem.startswith(_LINEAR_LISTS)):
        return problem or "fused expert stacks"
    leaf = chain[0]
    E = leaf.__dict__.get("num_experts")
    if isinstance(ep, int) and (E % ep or E < ep):
        return f"{E} experts do not split evenly over {ep} ranks"
    lists = _linear_lists(leaf)
    if any(p is not None for p in leaf._parameters.values()):
        return "expert parameters beside the per-expert lists"
    if any(_has_params(child) for name, child in leaf.named_children() if name not in lists):
        return "expert submodules beside the per-expert lists"
    if any(b is not None and b.numel() for b in leaf._buffers.values()):
        return "expert buffers"
    for name in lists:
        problem = list_shard_problem(_Holder(leaf._modules[name]), ep)
        if problem is not None:
            return f"{name}: {problem}"
    return None


class _Holder(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = experts


def _has_params(module):
    return any(True for _ in module.parameters())


def _linear_lists(leaf):
    E = leaf.__dict__.get("num_experts")
    return sorted(
        name
        for name, child in leaf.named_children()
        if type(child) is nn.ModuleList and len(child) == E and _has_params(child)
    )


class LocalExpertList(nn.ModuleList):
    """This rank's experts of one per-expert list: global names, local indexing 0 .. hi - lo."""

    _unsloth_global_expert_keys = True

    def __init__(self, local, lo):
        nn.Module.__init__(self)
        for e, module in local.items():
            self.add_module(str(e), module)
        self.__dict__["_lo"] = lo
        self.__dict__["_count"] = len(local)

    def __len__(self):
        return self.__dict__["_count"]

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        j = operator.index(i)
        if j < 0:
            j += len(self)
        if not 0 <= j < len(self):
            raise IndexError(f"local expert index {j} out of range ({len(self)} on this rank)")
        return self._modules[str(self.__dict__["_lo"] + j)]

    def __iter__(self):
        lo = self.__dict__["_lo"]
        return iter([self._modules[str(lo + j)] for j in range(len(self))])

    def __setitem__(self, i, module):
        raise TypeError("Unsloth: expert parallel expert lists are fixed")

    def append(self, module):
        raise TypeError("Unsloth: expert parallel expert lists are fixed")


def shard_fused_lists(block, mesh, rank, ep):
    """Shard `block`'s per-expert lists to [lo, hi) and route its fused call through ep_native's exchange."""
    epn, tpn = _ep(), _tpn()
    leaf = block._modules["experts"]
    E = leaf.num_experts
    per_rank = E // ep
    lo, hi = rank * per_rank, (rank + 1) * per_rank
    shard = _ListShard(block, lo, hi, per_rank, E, mesh.get_group(), ep)
    for name in _linear_lists(leaf):
        full = leaf._modules[name]
        _refuse_remote_local([full[e] for e in range(lo, hi)], block)
    for name in _linear_lists(leaf):
        full = leaf._modules[name]
        local = LocalExpertList({e: full[e] for e in range(lo, hi)}, lo)
        local.__dict__["_template"] = _meta_copy(full[lo])
        leaf._modules[name] = local
        shard.containers.append(local)
        for p in local.parameters():
            setattr(p, epn._EP, (LIST, E, ep))
    exchange = tpn._ExpertShard(block, leaf, lo, hi, per_rank, E, mesh.get_group())
    exchange.ep, exchange.calls = ep, 0
    exchange.full_attrs.append((leaf, "num_experts", E))
    leaf.num_experts = per_rank
    prior = leaf.__dict__.get("forward", leaf.forward)
    leaf.forward = epn._ep_experts_forward(exchange, prior)
    block.__dict__["_unsloth_tp_expert_shard"] = exchange
    block.__dict__[_SHARD] = shard
    shard.exchange = exchange
    _drop_load_guard(block)
    return exchange


def _logical_shape(p):
    remote = getattr(p, _REMOTE, None)
    if remote is not None:
        return remote[0]
    shape = getattr(getattr(p, "quant_state", None), "shape", None)
    return tuple(shape) if shape is not None else tuple(p.shape)


# Loading only this rank's experts (tp_native.expert_parallel_loading): another rank's expert keeps meta parameters
# tagged (logical shape, (experts, ep, rank)); apply_expert_parallel drops them.
_REMOTE = "_unsloth_ep_remote_expert"


def load_remote_experts(block, ep, rank):
    """Other ranks' expert modules under the shard_expert_list / shard_fused_lists split, else None."""
    if not isinstance(ep, int) or ep < 2 or _tpn().expert_shard_problem(block, ep) is None:
        return None
    if list_shard_problem(block, ep) is None:
        experts = block._modules["experts"]
        E = len(experts)
        per = -(-E // ep)
        lo = min(rank * per, E)
        hi = min(lo + per, E)
        return [experts[e] for e in range(E) if not lo <= e < hi]
    if fused_list_problem(block, ep) is None:
        leaf = block._modules["experts"]
        E = leaf.num_experts
        lo, hi = rank * (E // ep), (rank + 1) * (E // ep)
        return [
            leaf._modules[n][e] for n in _linear_lists(leaf) for e in range(E) if not lo <= e < hi
        ]
    return None


def _keep_meta(
    module,
    fn,
    recurse = True,
):
    return module


def keep_remote_on_meta(expert):
    expert.__dict__["_apply"] = functools.partial(_keep_meta, expert)


def remote_stand_in(p, tag):
    """A meta parameter in place of `p` (another rank's expert). A bitsandbytes 4-bit weight stays a packed
    Params4bit with a meta QuantState (PEFT reads quant_type, Unsloth sets quant_state.dtype after the load); its
    logical shape is kept for list_shard_problem."""
    shape = _logical_shape(p)
    new = None
    if _tpn()._is_params4bit(p):
        try:
            new = _params4bit_stand_in(p, shape)
        except Exception:
            new = None
    if new is None:
        new = nn.Parameter(
            torch.empty(p.shape, dtype = p.dtype, device = "meta"), requires_grad = p.requires_grad
        )
    if hasattr(p, "_is_hf_initialized"):
        new._is_hf_initialized = p._is_hf_initialized
    setattr(new, _REMOTE, (shape, tag))
    return new


def _params4bit_stand_in(p, shape):
    from bitsandbytes.functional import QuantState

    state = getattr(p, "quant_state", None)
    dtype = getattr(state, "dtype", None) or (
        p.dtype if p.dtype.is_floating_point else torch.float16
    )
    blocksize = getattr(p, "blocksize", None) or 64
    numel = 1
    for n in shape:
        numel *= n
    storage = getattr(p, "quant_storage", torch.uint8)
    packed = torch.empty((-(-numel // 2) // storage.itemsize, 1), dtype = storage, device = "meta")
    meta_state = QuantState(
        absmax = torch.empty(-(-numel // blocksize), dtype = torch.float32, device = "meta"),
        shape = torch.Size(shape),
        code = torch.empty(16, dtype = torch.float32, device = "meta"),
        blocksize = blocksize,
        quant_type = getattr(p, "quant_type", "nf4"),
        dtype = dtype,
    )
    return type(p)(
        packed,
        requires_grad = False,
        quant_state = meta_state,
        blocksize = blocksize,
        compress_statistics = getattr(p, "compress_statistics", True),
        quant_type = getattr(p, "quant_type", "nf4"),
        quant_storage = storage,
        module = getattr(p, "module", None),
        bnb_quantized = True,
    )


def is_remote(p):
    return getattr(p, _REMOTE, None) is not None


def replace_remote(module, pname, tag):
    """module.<pname> becomes another rank's meta stand-in; a Linear4bit's own quant_state reference goes too."""
    old = module._parameters[pname]
    module._parameters[pname] = new = remote_stand_in(old, tag)
    state = getattr(old, "quant_state", None)
    if state is not None and module.__dict__.get("quant_state") is state:
        module.__dict__["quant_state"] = new.quant_state


def check_remote_layout(model, ep, rank, ranks):
    """Before apply_expert_parallel changes anything: every stand-in was loaded for this ep mesh and rank."""
    for name, p in model.named_parameters():
        tag = getattr(p, _REMOTE, None)
        if tag is not None and tuple(tag[1][1:]) != (ep, rank, tuple(ranks)):
            raise ValueError(
                f"Unsloth: expert_parallel_loading kept only some experts ({name}) for ep {tag[1][1]} rank "
                f"{tag[1][2]} of ranks {tag[1][3]}, but apply_expert_parallel shards over ep {ep} rank {rank} of "
                f"ranks {tuple(ranks)}; load and shard with the same ep mesh."
            )


def _refuse_remote_local(lst_items, block):
    if any(is_remote(p) for m in lst_items for p in m.parameters()):
        raise ValueError(
            f"Unsloth: expert_parallel_loading kept other experts of {type(block).__name__} on this rank than "
            "apply_expert_parallel shards to it; load and shard with the same ep mesh."
        )


def _drop_load_guard(block):
    guard = block.__dict__.pop("_unsloth_tp_load_guard", None)  # set by expert_parallel_loading
    if guard is not None:
        guard.remove()


class _ListShard:
    """This rank owns global experts [lo, hi) of one block."""

    def __init__(self, block, lo, hi, per_rank, experts, group, ep):
        self.block, self.lo, self.hi, self.per_rank = block, lo, hi, per_rank
        self.experts, self.group, self.ep = experts, group, ep
        self.calls = 0  # block forwards that went through the exchange (engagement count)
        self.containers = []  # modules holding this rank's experts under their global indices
        # Children pass 1 may skip, per block.training; learned by _learn_skips.
        self.skip = {}


class _Proxy:
    """experts[e] inside the block forward: records (pass 1) or replays (pass 2)."""

    __slots__ = ("_owner", "_e")

    def __init__(self, owner, e):
        self._owner, self._e = owner, e

    def __call__(self, x, *args, **kwargs):
        if args or kwargs:
            raise RuntimeError(
                f"Unsloth: expert parallel calls each expert with its rows only, but "
                f"{self._owner._block_name} passed {len(args)} more arguments / {sorted(kwargs)}."
            )
        return self._owner._call(self._e, x)

    def __getattr__(self, name):
        raise RuntimeError(
            f"Unsloth: under expert parallel this rank holds only some experts, but {self._owner._block_name}.forward "
            f"read `{name}` of expert {self._e} directly (a fused or stacked path) instead of calling it. Train "
            "without expert parallel, or with a layout whose block calls each expert."
        )


class ExpertList(nn.Module):
    """This rank's experts under global indices; len() covers all E, outside the forward only local ones index."""

    _unsloth_global_expert_keys = (
        True  # child names are global expert indices (portable_resume orders by them)
    )

    def __init__(self, local, experts, block_name):
        super().__init__()
        for e, module in local.items():
            self.add_module(str(e), module)
        self.num_experts = experts
        self._block_name = block_name
        self._mode = None  # None, ("record", calls) or ("replay", outputs, cursor)

    def __len__(self):
        return self.num_experts

    def _index(self, i):
        e = operator.index(i)
        if e < 0:
            e += self.num_experts
        if not 0 <= e < self.num_experts:
            raise IndexError(f"expert index {e} out of range ({self.num_experts} experts)")
        return e

    def __getitem__(self, i):
        if isinstance(i, slice):
            raise TypeError("Unsloth: expert parallel experts cannot be sliced")
        e = self._index(i)
        if self._mode is not None:
            return _Proxy(self, e)
        module = self._modules.get(str(e))
        if module is None:
            raise IndexError(f"Unsloth: expert {e} is held by another expert parallel rank")
        return module

    def __iter__(self):
        if self._mode is not None:
            return iter([_Proxy(self, e) for e in range(self.num_experts)])
        return iter(self._modules.values())

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "Unsloth: expert parallel experts are called one by one (experts[e](rows))."
        )

    def _call(self, e, x):
        mode = self._mode
        if mode[0] == "record":
            mode[1].append((e, x))
            return torch.zeros_like(x)
        outputs, cursor = mode[1], mode[2]
        if cursor[0] >= len(outputs):
            raise RuntimeError(_replay_message(self, e, x, None))
        want, y = outputs[cursor[0]]
        if want != e or y.shape[0] != x.reshape(-1, x.shape[-1]).shape[0]:
            raise RuntimeError(_replay_message(self, e, x, (want, y.shape[0])))
        cursor[0] += 1
        return y.reshape(*x.shape[:-1], y.shape[-1])


def _replay_message(lst, e, x, recorded):
    return (
        f"Unsloth: expert parallel replays {lst._block_name}'s expert calls in the order the first pass made them, "
        f"but the second pass called expert {e} with {x.shape[0]} rows where the first called "
        f"{recorded}. Its routing is not deterministic for one input (a sampling router?): train it without "
        "expert parallel."
    )


def _rng_get(devices):
    return torch.get_rng_state(), [torch.cuda.get_rng_state(d) for d in devices]


def _rng_set(state, devices):
    torch.set_rng_state(state[0])
    for d, s in zip(devices, state[1]):
        torch.cuda.set_rng_state(s, d)


def _rng_devices(args, kwargs):
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
            return [value.device]
    return []


def _detach_inplace(value):
    # The first pass must not edit the caller's tensors (Mixtral's router jitter multiplies hidden_states in place).
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.clone()
    return value


def _exchange(shard, calls, like):
    """Exchange recorded rows to owners, run experts, return one output per call in order. Collective."""
    import torch.distributed as dist

    epn = _ep()
    group, ep, per_rank = shard.group, shard.ep, shard.per_rank
    width = calls[0][1].shape[-1] if calls else like.shape[-1]
    rows = [x.reshape(-1, x.shape[-1]) for _, x in calls]
    owners = [min(e // per_rank, ep - 1) for e, _ in calls]
    order = sorted(
        range(len(calls)), key = lambda i: owners[i]
    )  # stable: call order within a destination
    send = [0] * ep
    for i in order:
        send[owners[i]] += rows[i].shape[0]
    flag = int(not calls and like.reshape(-1, like.shape[-1]).shape[0] > 0)
    # If any rank's experts train, every rank's return exchange must be in autograd, even with no local gradient.
    lst = shard.block._modules["experts"]
    trains = int(torch.is_grad_enabled() and any(p.requires_grad for p in lst.parameters()))
    info = torch.tensor(
        [[n, flag, trains] for n in send], dtype = torch.int64, device = like.device
    ).reshape(-1)
    got = torch.empty_like(info)
    dist.all_to_all_single(got, info, group = group)
    got = got.view(ep, 3).tolist()
    recv = [n for n, _, _ in got]
    trains = trains or any(t for _, _, t in got)
    if any(f for _, f, _ in got) or flag:
        raise RuntimeError(
            f"Unsloth: {shard.block.__class__.__name__}.forward computed its experts without calling them one by "
            "one (a fused or decode-only path, or unsloth_zoo's grouped MoE forward), which under expert parallel "
            f"holds only this rank's experts. Disable that path ({_ZOO_GROUPED}=0 for the zoo's) or train without "
            "expert parallel."
        )
    if calls:
        x_send = torch.cat([rows[i] for i in order])
        e_send = torch.cat(
            [
                torch.full((rows[i].shape[0],), calls[i][0], dtype = torch.int64, device = like.device)
                for i in order
            ]
        )
    else:
        x_send = like.reshape(-1, width)[:0]
        e_send = torch.zeros(0, dtype = torch.int64, device = like.device)
    got_e = epn._a2a(e_send, send, recv, group)
    (got_x,) = epn._Exchange.apply(send, recv, group, x_send)
    lo, hi = shard.lo, shard.hi
    local = got_e - lo
    perm = torch.argsort(local, stable = True)
    counts = torch.bincount(local, minlength = hi - lo).tolist() if local.numel() else [0] * (hi - lo)
    tie = got_x[
        :0
    ].sum()  # keeps this rank's backward exchanges in the graph when it receives nothing
    grouped = _grouped_locals(shard, lst, got_x, perm, counts)
    pieces, start = [], 0
    for j, count in enumerate(counts if grouped is None else ()):
        if count == 0:
            continue  # no rank routed to it: its gradients stay None, as in one process
        expert = lst._modules[str(lo + j)]
        pieces.append(
            (
                perm[start : start + count],
                expert(got_x.index_select(0, perm[start : start + count])),
            )
        )
        start += count
    if grouped is not None:
        y_got = grouped
    elif pieces:
        out_width = pieces[0][1].shape[-1]
        y_got = pieces[0][1].new_zeros((got_x.shape[0], out_width))
        for index, y in pieces:
            y_got = y_got.index_add(0, index, y)
    else:
        y_got = got_x.new_zeros((got_x.shape[0], width))
    y_got = y_got + tie.to(y_got.dtype)
    if trains and not y_got.requires_grad:
        y_got = y_got.detach().requires_grad_()
    (y_back,) = epn._Exchange.apply(recv, send, group, y_got)
    outputs = [None] * len(calls)
    start = 0
    for i in order:
        n = rows[i].shape[0]
        outputs[i] = (calls[i][0], y_back[start : start + n])
        start += n
    return outputs, y_back


# Local experts as one grouped GEMM per projection instead of a module call each: the received rows are already
# sorted by local expert. The per-expert MLPs whose forward is down(act(gate(x)) * up(x)) (transformers 4.x).
_GROUPED_MLPS = {
    "Qwen3MoeMLP": ("gate_proj", "up_proj", "down_proj"),
    "Qwen2MoeMLP": ("gate_proj", "up_proj", "down_proj"),
    "OlmoeMLP": ("gate_proj", "up_proj", "down_proj"),
    "MixtralBlockSparseTop2MLP": ("w1", "w3", "w2"),
    "DeepseekV3MLP": ("gate_proj", "up_proj", "down_proj"),
    "Glm4MoeMLP": ("gate_proj", "up_proj", "down_proj"),
    "Qwen3NextMLP": ("gate_proj", "up_proj", "down_proj"),
    "HunYuanMoEV1MLP": ("gate_proj", "up_proj", "down_proj"),
}
_GROUPED_ENV = "UNSLOTH_EP_GROUPED_EXPERTS"


def _grouped_spec(shard, lst, counts):
    """(the experts that received rows as one ModuleList, projection names), or a reason. Only routed experts go in,
    so an expert with no rows keeps unset gradients, as in the loop. The checks run every call (hooks and patched
    forwards can appear later); the ModuleList, which the zoo kernels cache their dequant tables on, is reused."""
    lo, hi = shard.lo, shard.hi
    if hi <= lo or os.environ.get(_GROUPED_ENV, "1") == "0":
        return "no local experts" if hi <= lo else f"{_GROUPED_ENV}=0"
    experts = [lst._modules[str(lo + j)] for j, count in enumerate(counts) if count]
    kind = type(experts[0])
    names = _GROUPED_MLPS.get(kind.__name__)
    if (
        names is None
        or kind.forward.__module__ != kind.__module__
        or not callable(getattr(experts[0], "act_fn", None))
    ):
        return f"{kind.__name__} is not a known expert MLP"
    if any(
        type(m) is not kind or "forward" in m.__dict__ or m._forward_hooks or m._forward_pre_hooks
        for m in experts
    ):
        return "experts of different types, or with hooks"
    if _fake_quantized(experts, names):
        return "fake-quantized (QAT) expert projections"
    if _projection_hooks(experts, names):
        return "hooks on expert projections"
    cache = shard.__dict__.setdefault("_grouped", {})
    key = tuple(id(m) for m in experts)
    hit = cache.get(key)
    if hit is None:
        if len(cache) >= 8:
            cache.pop(next(iter(cache)))
        cache[key] = hit = nn.ModuleList(experts)  # not attached to the model
    return hit, names


def _fake_quantized(experts, names):
    # QAT: the grouped builders read raw weights and would skip the projections' fake quantizers.
    from ..kernels.fast_lora import _has_activation_fake_quantizer
    for expert in experts:
        for name in names:
            proj = expert._modules[name]
            base = proj._modules.get("base_layer", proj)
            if getattr(
                base, "weight_fake_quantizer", None
            ) is not None or _has_activation_fake_quantizer(proj):
                return True
    return False


def _projection_hooks(experts, names):
    # The grouped builders read the weights without calling the projections, so their hooks would not run.
    for expert in experts:
        for name in names:
            for m in expert._modules[name].modules():
                if m._forward_hooks or m._forward_pre_hooks:
                    return True
    return False


def _grouped_locals(shard, lst, got_x, perm, counts):
    """This rank's experts over the received rows with unsloth_zoo's grouped GEMMs (bitsandbytes 4-bit or 16-bit
    frozen bases, one LoRA adapter), rows in got_x order; None keeps the per-expert loop (reason on the shard)."""
    found = (
        f"{got_x.device.type} {got_x.dtype} rows"
        if not got_x.is_cuda
        or got_x.dtype not in (torch.float16, torch.bfloat16)
        or got_x.shape[0] == 0
        else _grouped_spec(shard, lst, counts)
    )
    if isinstance(found, str):
        shard.grouped_declined = found
        return None
    experts, names = found
    try:
        from unsloth_zoo.temporary_patches import moe_grouped_modulelist as zoo
        from unsloth_zoo.temporary_patches.moe_utils import _grouped_mm_supported_on
    except Exception as exc:
        shard.grouped_declined = f"unsloth_zoo grouped MoE unavailable: {type(exc).__name__}"
        return None
    if not _grouped_mm_supported_on(got_x.device):
        shard.grouped_declined = "grouped GEMM unsupported on this GPU"
        return None
    spec = (*names, None)
    dtype = got_x.dtype
    lora = zoo._experts_grouped_state(experts, spec, got_x.device, dtype)
    if not isinstance(lora, dict):
        shard.grouped_declined = lora
        return None
    projs = zoo._expert_projs(experts, spec)
    routed = [count for count in counts if count]
    offsets = torch.tensor(routed, dtype = torch.int64).cumsum(0).to(torch.int32).to(got_x.device)
    x = got_x.index_select(0, perm)
    gate_up = zoo._grouped_expert_gemm(
        x, offsets, lambda: zoo._build_gate_up_stack(experts, spec, dtype, projs), True
    )
    gate, up = gate_up.chunk(2, dim = -1)
    if lora["gate"] is not None:
        gate = gate + zoo._lora_delta(x, offsets, projs.gate, lora["gate"], dtype)
    if lora["up"] is not None:
        up = up + zoo._lora_delta(x, offsets, projs.up, lora["up"], dtype)
    inter = experts[0].act_fn(gate) * up
    down = zoo._grouped_expert_gemm(
        inter, offsets, lambda: zoo._build_down_stack(experts, spec, dtype, projs), True
    )
    if lora["down"] is not None:
        down = down + zoo._lora_delta(inter, offsets, projs.down, lora["down"], dtype)
    shard.grouped_calls = getattr(shard, "grouped_calls", 0) + 1
    return down.new_zeros((got_x.shape[0], down.shape[-1])).index_copy(0, perm, down)


class _Zeros(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x, *args, **kwargs):
        return x.new_zeros(x.shape, dtype = self.dtype)


@contextlib.contextmanager
def _skipping(block, skip):
    held = {name: block._modules[name] for name, _ in skip}
    try:
        for name, dtype in skip:
            block._modules[name] = _Zeros(dtype)
        yield
    finally:
        block._modules.update(held)


# The 4.x shared expert children, whose output is only added after routing; a router of width H is never one.
_SHARED = ("shared_expert", "shared_experts", "shared_mlp")


def _draws_rng(block, skip):
    # A dropout active now (PEFT set_adapter, train mode) must run in pass 1 too: the RNG ends after pass 1.
    return any(
        isinstance(m, nn.Dropout) and m.training and m.p > 0
        for name, _ in skip
        for m in block._modules[name].modules()
    )


@contextlib.contextmanager
def _watch_shapes(block):
    """Shared expert children mapping H to H: {name: output dtype}."""
    seen, handles = {}, []

    def hook(name):
        def record(module, args, output):
            x = next((a for a in args if isinstance(a, torch.Tensor)), None)
            same = isinstance(output, torch.Tensor) and x is not None and output.shape == x.shape
            dtype = seen.get(name)
            if not same or (dtype is not None and dtype != output.dtype):
                seen[name] = False
            elif dtype is None:
                seen[name] = output.dtype

        return record

    for name in _SHARED:
        child = block._modules.get(name)
        if child is not None:
            handles.append(child.register_forward_hook(hook(name)))
    found = {}
    try:
        yield found
    finally:
        for h in handles:
            h.remove()
        found.update({n: d for n, d in seen.items() if isinstance(d, torch.dtype)})


def _same_calls(a, b):
    return len(a) == len(b) and all(
        ea == eb and xa.shape == xb.shape and torch.equal(xa, xb)
        for (ea, xa), (eb, xb) in zip(a, b)
    )


def _same_rng(a, b):
    return torch.equal(a[0], b[0]) and all(torch.equal(x, y) for x, y in zip(a[1], b[1]))


def _pass1(shard, prior, args, kwargs, calls):
    lst = shard.block._modules["experts"]
    lst._mode = ("record", calls)
    try:
        prior(
            *(_detach_inplace(a) for a in args),
            **{k: _detach_inplace(v) for k, v in kwargs.items()},
        )
    finally:
        lst._mode = None


def _learn_skips(shard, prior, args, kwargs, before, devices):
    """No-grad pass 1 trials from `before`: children are skippable only if expert calls and RNG draws stay bitwise
    equal. Restores the RNG to `before` so later passes (checkpoint recompute too) choose the same."""
    full, trial = [], []
    try:
        with torch.no_grad():
            _rng_set(before, devices)
            with _watch_shapes(shard.block) as candidates:
                _pass1(shard, prior, args, kwargs, full)
            skip = tuple(sorted(candidates.items()))
            if not skip:
                return ()
            drawn = _rng_get(devices)
            _rng_set(before, devices)
            with _skipping(shard.block, skip):
                _pass1(shard, prior, args, kwargs, trial)
            same = _same_rng(_rng_get(devices), drawn) and _same_calls(trial, full)
    except Exception:
        return ()
    finally:
        _rng_set(before, devices)
    return skip if same else ()


def _list_forward(shard, prior):
    @functools.wraps(prior)
    def forward(*args, **kwargs):
        lst = shard.block._modules["experts"]
        like = next(
            (
                v
                for v in (*args, *kwargs.values())
                if isinstance(v, torch.Tensor) and v.is_floating_point()
            ),
            None,
        )
        if like is None:
            raise RuntimeError(
                "Unsloth: expert parallel found no hidden states in the MoE block call."
            )
        calls = []
        devices = _rng_devices(args, kwargs)
        # Both passes draw the same randomness; the RNG ends where one process would leave it.
        before = _rng_get(devices)
        mode = bool(shard.block.training)
        if mode not in shard.skip:
            shard.skip[mode] = _learn_skips(shard, prior, args, kwargs, before, devices)
        skip = shard.skip[mode]
        if skip and _draws_rng(shard.block, skip):
            skip = ()
        # Pass 1 output is discarded, so skip children that cannot change the expert calls.
        with _skipping(shard.block, skip):
            _pass1(shard, prior, args, kwargs, calls)
        outputs, y_back = _exchange(shard, calls, like)
        shard.calls += 1
        after = _rng_get(devices)
        _rng_set(before, devices)
        cursor = [0]
        lst._mode = ("replay", outputs, cursor)
        try:
            result = prior(*args, **kwargs)
        finally:
            lst._mode = None
            _rng_set(after, devices)
        if cursor[0] != len(outputs):
            raise RuntimeError(
                f"Unsloth: expert parallel recorded {len(outputs)} expert calls of {type(shard.block).__name__} but "
                f"the second pass made {cursor[0]}; its routing is not deterministic for one input."
            )
        if not outputs and torch.is_grad_enabled() and y_back.requires_grad:
            # Nothing routed from this rank: tie the (empty) return exchange in, so its backward runs here too.
            if isinstance(result, tuple):
                result = (result[0] + y_back[:0].sum().to(result[0].dtype), *result[1:])
            else:
                result = result + y_back[:0].sum().to(result.dtype)
        return result

    forward._unsloth_ep_expert_list = True
    return _ep()._compiler_disable(forward)


def shard_expert_list(block, mesh, rank, ep):
    """Shard `block`'s experts to [lo, hi) and route its forward through record / exchange / replay."""
    epn = _ep()
    experts = block._modules["experts"]
    E = len(experts)
    per_rank = -(-E // ep)
    lo = min(rank * per_rank, E)
    hi = min(lo + per_rank, E)
    local = {e: experts[e] for e in range(lo, hi)}
    _refuse_remote_local(local.values(), block)
    lst = ExpertList(local, E, type(block).__name__)
    lst.__dict__["_template"] = _meta_copy(experts[lo if lo < hi else 0])
    block._modules["experts"] = lst
    shard = _ListShard(block, lo, hi, per_rank, E, mesh.get_group(), ep)
    shard.containers.append(lst)
    for p in lst.parameters():
        setattr(p, epn._EP, (LIST, E, ep))
    prior = block.__dict__.get("forward", block.forward)
    block.forward = _list_forward(shard, prior)
    block.__dict__[_SHARD] = shard
    _drop_load_guard(block)
    return shard


def gather_list_state(
    model,
    group,
    dst = None,
    adapter = False,
):
    """{name: tensor} of every expert's parameters (adapter only if `adapter`) from all ep ranks; only `dst`
    receives when given, None elsewhere. Collective."""
    import torch.distributed as dist

    epn = _ep()
    names = {id(m): n for n, m in model.named_modules()}
    receives = dst is None or dist.get_rank() == dst
    merged = {} if receives else None
    for block, shard, lst in _containers(model):
        prefix = names[id(lst)]
        template = lst.__dict__["_template"]
        entries = [
            (j, suffix, meta)
            for j in range(shard.per_rank)
            for suffix, meta in template.named_parameters()
            if not adapter or meta.requires_grad or "lora_" in suffix
        ]
        if not adapter and any(_tpn()._is_params4bit(p) for p in lst.parameters()):
            raise NotImplementedError(
                "Unsloth: save_pretrained of the whole model under expert parallel would gather 4-bit expert "
                "weights; save the adapter (the PEFT model's save_pretrained or trainer.save_model)."
            )
        device = next((p.device for p in lst.parameters()), None) or epn._group_device(group)
        for dtype in sorted({m.dtype for _, _, m in entries}, key = str):
            items = [x for x in entries if x[2].dtype == dtype]
            local = []
            for j, suffix, meta in items:
                e = shard.lo + j
                if e < shard.hi:
                    local.append(
                        lst._modules[str(e)].get_parameter(suffix).detach().reshape(-1).to(device)
                    )
                else:
                    local.append(torch.zeros(meta.numel(), dtype = dtype, device = device))
            flat = torch.cat(local) if local else torch.zeros(0, dtype = dtype, device = device)
            if dst is None:
                parts = [torch.empty_like(flat) for _ in range(shard.ep)]
                dist.all_gather(parts, flat, group = group)
            else:
                parts = [torch.empty_like(flat) for _ in range(shard.ep)] if receives else None
                dist.gather(flat, parts, dst = dst, group = group)
            if not receives:
                continue
            for s, part in enumerate(parts):
                offset = 0
                for j, suffix, meta in items:
                    e, n = s * shard.per_rank + j, meta.numel()
                    if e < shard.experts:
                        value = part[offset : offset + n].view(meta.shape)
                        merged[f"{prefix}.{e}.{suffix}"] = (
                            value.cpu() if dst is not None else value.clone()
                        )
                    offset += n
    return merged


def list_blocks(model):
    return [m for m in model.modules() if is_listed(m)]


def _containers(model):
    for block in list_blocks(model):
        shard = block.__dict__[_SHARD]
        for lst in shard.containers:
            yield block, shard, lst


_HOOK_DICTS = (
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)


def _meta_copy(module):
    """Meta copy of `module` standing in for another rank's expert, so PEFT >= 0.21 saves keep its keys."""
    memo = {}
    for t in itertools.chain(module.parameters(), module.buffers()):
        meta = torch.empty(t.shape, dtype = t.dtype, device = "meta")
        memo[id(t)] = (
            nn.Parameter(meta, requires_grad = t.requires_grad)
            if isinstance(t, nn.Parameter)
            else meta
        )
    for m in module.modules():
        for name in _HOOK_DICTS:
            hooks = m.__dict__.get(name)
            if hooks is not None:
                memo[id(hooks)] = collections.OrderedDict()
    return copy.deepcopy(module, memo)


@contextlib.contextmanager
def remote_experts_for_save(model):
    """On the writing rank during save_pretrained: other ranks' experts present as meta stand-ins."""
    added = []
    try:
        for _, shard, lst in _containers(model):
            template = lst.__dict__["_template"]
            for e in range(shard.experts):
                if str(e) not in lst._modules:
                    lst._modules[str(e)] = _meta_copy(template)
                    added.append((lst, str(e)))
        yield
    finally:
        for lst, key in added:
            lst._modules.pop(key, None)
