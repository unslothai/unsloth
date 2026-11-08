# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Opt-in fused NF4 Llama decode: UNSLOTH_DECODE_FUSED_NF4=1 (tuned GPUs) or =force.
A failed check keeps the regular path; projection weights retain their original storage."""

import os
import weakref

import torch

from . import _decode_graphs
from ..kernels import utils as KU

try:
    from ..kernels import nf4_decode_fused as F
except Exception:
    # torch < 2.6 has no torch.library.triton_op (kernels/nf4.py): keep the regular decode path.
    F = None

__all__ = ["requested", "decode_forward", "release", "STATS"]

ENV = "UNSLOTH_DECODE_FUSED_NF4"
STATS = {
    "steps": 0,
    "layers": 0,
    "graph_records": 0,
    "plans": 0,
    "refused": None,
}
_PLAN, _BATCH_PLAN, _HOMES = _decode_graphs.FUSED_NF4_ATTRS


def _backend_ok():
    # The regular NF4 path's own switches: UNSLOTH_BNB_TRITON=0, or Triton NF4 kernels that are
    # unavailable or failed (utils._nf4_kernels_failed), keep bitsandbytes for every 4bit op.
    return os.environ.get("UNSLOTH_BNB_TRITON", "1") != "0" and bool(KU._USE_NF4_KERNELS)


def requested():
    return os.environ.get(ENV, "0") in ("1", "force") and _backend_ok()


def graph_key():
    """Part of a captured decode step's key: fused and regular steps never replay each other."""
    return os.environ.get(ENV, "0") if requested() else "0"


def release(model):
    """Drops fused plans and any caches left by an earlier implementation."""
    for module in model.modules():
        d = module.__dict__
        d.pop(_PLAN, None)
        d.pop(_BATCH_PLAN, None)
        d.pop(_HOMES, None)
    _decode_graphs.invalidate(model)


class _ApplyReleasing:
    def __init__(self, root, module):
        self.root_ref = weakref.ref(root) if root is not None else lambda: None
        self.module_ref = weakref.ref(module)

    def __call__(self, fn, *args, **kwargs):
        root, module = self.root_ref(), self.module_ref()
        if root is not None and (
            _PLAN in root.__dict__ or _BATCH_PLAN in root.__dict__ or _HOMES in root.__dict__
        ):
            release(root)
        return type(module)._apply(module, fn, *args, **kwargs)

    def __reduce__(self):
        # Copies and pickle must bind the hook to the restored modules. Only the root's own hook keeps
        # its owner: pickling a submodule must not serialize the whole model through it.
        root, module = self.root_ref(), self.module_ref()
        return type(self), (root if root is module else None, module)

    def __deepcopy__(self, memo):
        return type(self)(memo.get(id(self.root_ref())), memo.get(id(self.module_ref())))


def _track_moves(model):
    # Reloaded adapters add modules after the root has already been tracked.
    for module in model.modules():
        binding = module.__dict__.get("_apply")
        if binding is None or (
            isinstance(binding, _ApplyReleasing)
            and (binding.root_ref() is not model or binding.module_ref() is not module)
        ):
            module.__dict__["_apply"] = _ApplyReleasing(model, module)


class _Refused:
    __slots__ = ("reason",)

    def __init__(self, reason):
        self.reason = reason


class _Layer:
    __slots__ = ("attn", "sites", "ln1", "ln2", "head_dim", "n_heads", "n_kv_heads", "rope")


class _Plan:
    pass


def _norm_eps(norm):
    eps = getattr(norm, "variance_epsilon", None)
    return getattr(norm, "eps", None) if eps is None else eps


def _scale_key(v):
    # PEFT hotswap fill_()s a tensor scale in place: key on its version counter (no sync), and hold
    # the tensor so its id cannot be reused. Inference tensors have no version counter.
    if not isinstance(v, torch.Tensor):
        return v
    return (id(v), None if v.is_inference() else v._version, v)


def _adapter_key(module):
    active = getattr(module, "active_adapters", None)
    if active is None:
        active = getattr(module, "active_adapter", None)
    if isinstance(active, list):
        active = tuple(active)
    scaling = getattr(module, "scaling", None)
    if isinstance(scaling, dict) and isinstance(active, tuple):
        scaling = tuple(_scale_key(v) for v in (scaling.get(a) for a in active))
    else:
        scaling = None
    weights = None
    lora_A = getattr(module, "lora_A", None)
    lora_B = getattr(module, "lora_B", None)
    if lora_A is not None and lora_B is not None and isinstance(active, tuple):
        weights = []
        for a in active:
            if a in lora_A and a in lora_B:
                A, B = lora_A[a].weight, lora_B[a].weight
                weights.append((id(A), id(B), A.shape, B.shape))
        weights = tuple(weights)
    return (
        getattr(module, "disable_adapters", None),
        getattr(module, "merged", None),
        active,
        scaling,
        weights,
    )


def _adapter_state(modules):
    return tuple(_adapter_key(m) for m in modules)


def _site_watch(projs):
    return [p for p in projs if hasattr(p, "lora_A")]


def _plan_stale(plan):
    """Adapter state or module bindings changed since this plan was built."""
    return (
        _adapter_state(plan.watch) != plan.adapter_state
        or any(
            getattr(owner, name, None) is not module
            for owner, name, module in getattr(plan, "bindings", ())
        )
        or any(
            getattr(module, name, None) is not tensor
            for module, name, tensor in getattr(plan, "tensor_bindings", ())
        )
    )


def _same_tables(a, b):
    """Equal copied NF4 tables (codes, absmax2 codes, nested offsets): a re-quantized weight can reuse every
    address of the old one, so equal pointers alone do not mean equal values. Once per sequence."""
    differs = []
    for ea, eb in zip(a.layers, b.layers):
        for kind, sa in ea.sites.items():
            sb = eb.sites[kind]
            if sa.nested != sb.nested:
                return False
            for name in ("code", "code2", "seg_offset") if sa.nested else ("code",):
                ta, tb = getattr(sa, name), getattr(sb, name)
                if ta is tb:
                    continue
                if ta.shape != tb.shape or ta.device != tb.device:
                    return False
                differs.append((ta != tb).any())
    return not differs or not bool(torch.stack(differs).any())


def _reusable(plan, built):
    # Keep the previous plan (and the graphs that captured its buffers) only for identical weights and the same
    # attention / RoPE modules (a replaced block holding the same tensors still has its own cache and tables).
    return (
        built.signature == plan.signature
        and len(built.layers) == len(plan.layers)
        and all(a.attn is b.attn and a.rope is b.rope for a, b in zip(plan.layers, built.layers))
        and _same_tables(built, plan)
    )


def _signature(plan):
    return (
        (
            plan.device,
            plan.dtype,
            plan.hidden,
            plan.intermediate,
            plan.eps,
            plan.final_norm.data_ptr(),
        )
        + tuple((entry.ln1.data_ptr(), entry.ln2.data_ptr()) for entry in plan.layers)
        + tuple(
            (
                site.packed.data_ptr(),
                site.absmax.data_ptr(),
                site.absmax2.data_ptr(),
                tuple(
                    tuple(t.data_ptr() for t in fields if t is not None)
                    for fields in (getattr(site, "segments", None) or ())
                ),
                site.lora and tuple(t.data_ptr() for t in site.A3 + site.B3),
                site.lora and (site.rank, site.scaling),
                tuple(site.bounds),
            )
            for entry in plan.layers
            for site in entry.sites.values()
        )
    )


def _proj_parts(proj, K):
    """(W, quant_state, (A, B, S) or None) of a 4bit NF4 projection the kernel can read, else a str."""
    if (
        KU._has_non_lora_adapter(proj)
        or KU._has_multiple_active_adapters(proj)
        or KU._has_active_dora_adapter(proj)
        or KU._has_active_lora_bias(proj)
        or KU._has_packed_qweight(proj)
        or proj.__dict__.get("_unsloth_tp_local") is not None
    ):
        return "adapter or projection type"
    W, qs, A, B, S, bias = KU.get_lora_parameters_bias(proj)
    if bias is not None:
        return "bias"
    if qs is None or type(qs) is list or getattr(qs, "quant_type", None) != "nf4":
        return "not NF4"
    if (
        W.dtype != torch.uint8
        or not W.is_contiguous()
        or KU._quant_state_dtype(qs) == torch.float32
    ):
        return "NF4 storage"
    N, k = (int(v) for v in qs.shape)
    blocksize = int(qs.blocksize)
    if k != K or K % blocksize or blocksize & (blocksize - 1) or W.numel() * 2 != N * K:
        return "NF4 shape"
    if qs.state2 is not None:
        blocksize2 = int(qs.state2.blocksize)
        # Segments are concatenated: each must hold whole absmax2 blocks.
        if blocksize2 & (blocksize2 - 1) or (N * K // blocksize) % blocksize2:
            return "nested absmax layout"
    elif qs.absmax.dtype != torch.float32:
        return "absmax dtype"
    lora = None
    if A is not None:
        if A.dim() != 2 or A.shape[1] != K or B.shape != (N, A.shape[0]):
            return "LoRA shape"
        if (
            not (A.is_contiguous() and B.is_contiguous())
            or A.device != W.device
            or B.device != W.device
        ):
            return "LoRA layout"
        if isinstance(S, torch.Tensor):
            if S.is_inference():
                return "inference tensor LoRA scale"
            S = float(S)
        lora = (A, B, float(S))
    return W, qs, N, lora


def _site(
    model,
    key,
    kind,
    projs,
    K,
    batch_size = 1,
):
    parts = [_proj_parts(p, K) for p in projs]
    for part in parts:
        if isinstance(part, str):
            return part
    Ws = [part[0] for part in parts]
    states = [part[1] for part in parts]
    loras = [part[3] for part in parts]
    st0 = states[0]
    nested = st0.state2 is not None
    for qs in states[1:]:
        if (
            int(qs.blocksize) != int(st0.blocksize)
            or (qs.state2 is not None) != nested
            or (nested and int(qs.state2.blocksize) != int(st0.state2.blocksize))
            or qs.absmax.dtype != st0.absmax.dtype
            or (nested and qs.state2.absmax.dtype != st0.state2.absmax.dtype)
        ):
            return "segments differ"
    # The kernel reads every segment through raw pointers on the launch device.
    home = Ws[0].device
    for W, qs in zip(Ws, states):
        tensors = (W, qs.absmax) + ((qs.state2.absmax,) if nested else ())
        if any(t.device != home for t in tensors):
            return "several devices"
    if any(l is None for l in loras) != all(l is None for l in loras):
        return "LoRA on some projections only"
    lora = None
    if loras[0] is not None:
        r = loras[0][0].shape[0]
        if (
            r > 256
            or any(l[0].shape[0] != r or l[2] != loras[0][2] for l in loras)
            or len({l[0].dtype for l in loras}) != 1
            or len({l[1].dtype for l in loras}) != 1
        ):
            return "LoRA ranks / scales / dtypes differ"
        lora = [(l[0], l[1]) for l in loras]
    if "_apply" not in model.__dict__:
        _track_moves(model)
    if any(not torch.equal(qs.code, st0.code) for qs in states[1:]):
        return "NF4 tables differ"
    if nested and any(not torch.equal(qs.state2.code, st0.state2.code) for qs in states[1:]):
        return "absmax2 tables differ"
    packed, absmax = Ws[0], st0.absmax
    absmax2 = st0.state2.absmax if nested else None
    bounds, acc = [], 0
    for part in parts:
        acc += part[2]
        bounds.append(acc)
    seg_offset = None
    if nested:
        offsets = [
            qs.offset.reshape(()).float()
            if isinstance(qs.offset, torch.Tensor)
            else torch.tensor(float(qs.offset), device = packed.device)
            for qs in states
        ]
        offsets += [offsets[-1]] * (3 - len(offsets))
        seg_offset = torch.stack(offsets).to(packed.device).contiguous()
    site = F.FusedSite(
        kind,
        packed,
        absmax,
        st0.code.float().contiguous(),
        bounds,
        K,
        int(st0.blocksize),
        code2 = st0.state2.code.float().contiguous() if nested else None,
        absmax2 = absmax2,
        seg_offset = seg_offset,
        blocksize2 = int(st0.state2.blocksize) if nested else 256,
        lora = lora,
        scaling = loras[0][2] if lora is not None else 1.0,
        segments = tuple(
            (W, qs.absmax, qs.state2.absmax if nested else None) for W, qs in zip(Ws, states)
        ),
    )
    site.tensor_bindings = tuple(
        (getattr(proj, "base_layer", proj), "weight", W) for proj, W in zip(projs, Ws)
    )
    return site if batch_size == 1 else F.BatchSite(site, batch_size)


def _build(model, X):
    if F is None:
        return "fused NF4 kernels unavailable (torch < 2.6)"
    batch = X.shape[0]
    config = model.config
    if getattr(config, "model_type", None) != "llama":
        return "model type"
    if getattr(model, "vllm_engine", None) is not None:
        # vLLM owns its inference path and captured state.
        return "weights shared with vLLM (fast_inference)"
    device = X.device
    if device.type != "cuda" or torch.version.hip is not None:
        return "not NVIDIA CUDA"
    if X.dtype not in (torch.float16, torch.bfloat16):
        return "dtype"
    capability = torch.cuda.get_device_capability(device)
    if batch != 1 and capability != (7, 5) and os.environ.get(ENV) != "force":
        return "batch NF4 decode is tuned on T4; use force for other GPUs"
    # T4 fusion adds compilation cost without a measured gain over segmented graphs.
    if capability == (7, 5) and _decode_graphs.enabled() and os.environ.get(ENV) != "force":
        return "T4 uses regular CUDA graphs"
    if not F.supported_capability(capability) and os.environ.get(ENV) != "force":
        return (
            f"no tuned config for sm{capability[0]}{capability[1]} (UNSLOTH_DECODE_FUSED_NF4=force)"
        )
    if getattr(config, "hidden_act", "silu") != "silu":
        return "activation"
    layers = model.model.layers
    if getattr(layers, "_unsloth_block_swap", None) is not None or len(layers) == 0:
        return "block swap"
    hidden = config.hidden_size
    plan = _Plan()
    plan.layers = []
    plan.device = device
    plan.dtype = X.dtype
    plan.capability = capability
    watch, bindings, tensor_bindings = [], [], []
    norm = model.model.norm
    eps = _norm_eps(norm)
    for idx, layer in enumerate(layers):
        attn, mlp = layer.self_attn, layer.mlp
        # Replacing a whole attention / MLP block must invalidate the plan like its projections.
        bindings.append((layer, "self_attn", attn))
        bindings.append((layer, "mlp", mlp))
        if (
            getattr(attn, "q_norm", None) is not None
            or getattr(attn, "k_norm", None) is not None
            or attn.__dict__.get("_unsloth_tp_heads") is not None
            or getattr(attn, "rotary_emb", None) is None
        ):
            return "attention type"
        if (
            _norm_eps(layer.input_layernorm) != eps
            or _norm_eps(layer.post_attention_layernorm) != eps
        ):
            return "norm eps"
        for w in (layer.input_layernorm.weight, layer.post_attention_layernorm.weight):
            if w.device != device or w.shape != (hidden,) or not w.is_contiguous():
                return "norm weight"
        head_dim = attn.head_dim
        n_heads = config.num_attention_heads
        n_kv_heads = config.num_key_value_heads
        if head_dim % 2:
            return "head dim"
        for name in ("input_layernorm", "post_attention_layernorm"):
            module = getattr(layer, name)
            bindings.append((layer, name, module))
            tensor_bindings.append((module, "weight", module.weight))
            if hasattr(module, "modules_to_save"):  # a norm in modules_to_save
                watch.append(module)
        entry = _Layer()
        entry.attn = attn
        entry.ln1 = layer.input_layernorm.weight
        entry.ln2 = layer.post_attention_layernorm.weight
        entry.head_dim, entry.n_heads, entry.n_kv_heads = head_dim, n_heads, n_kv_heads
        entry.rope = attn.rotary_emb
        bindings.append((attn, "rotary_emb", entry.rope))
        sites = {}
        for kind, owner, names, K in (
            ("qkv", attn, ("q_proj", "k_proj", "v_proj"), hidden),
            ("o", attn, ("o_proj",), n_heads * head_dim),
            ("gate_up", mlp, ("gate_proj", "up_proj"), hidden),
            ("down", mlp, ("down_proj",), config.intermediate_size),
        ):
            projs = [getattr(owner, name, None) for name in names]
            if any(p is None for p in projs):
                return "projections"
            site = (
                _site(model, (idx, kind), kind, projs, K)
                if batch == 1
                else _site(model, (idx, kind), kind, projs, K, batch_size = batch)
            )
            if isinstance(site, str):
                return f"{kind}: {site}"
            watch.extend(_site_watch(projs))
            bindings.extend((owner, name, proj) for name, proj in zip(names, projs))
            tensor_bindings.extend(getattr(site, "tensor_bindings", ()))
            if site.packed.device != device:
                return "several devices"
            sites[kind] = site
        q_rows = n_heads * head_dim
        if sites["qkv"].bounds != [
            q_rows,
            q_rows + n_kv_heads * head_dim,
            q_rows + 2 * n_kv_heads * head_dim,
        ]:
            return "qkv shape"
        if sites["o"].N != hidden or sites["down"].N != hidden:
            return "o / down shape"
        if sites["gate_up"].bounds[0] * 2 != sites["gate_up"].N:
            return "gate / up shape"
        entry.sites = sites
        plan.layers.append(entry)
    if (
        norm.weight.device != device
        or norm.weight.shape != (hidden,)
        or not norm.weight.is_contiguous()
    ):
        return "final norm"
    bindings.append((model.model, "norm", norm))
    tensor_bindings.append((norm, "weight", norm.weight))
    if hasattr(norm, "modules_to_save"):
        watch.append(norm)
    plan.watch = watch
    plan.bindings = tuple(bindings)
    plan.tensor_bindings = tuple(tensor_bindings)
    plan.adapter_state = _adapter_state(watch)
    plan.eps = float(eps)
    plan.final_norm = norm.weight
    plan.hidden = hidden
    plan.intermediate = config.intermediate_size
    plan.batch_size = batch
    hidden_shape = hidden if batch == 1 else (batch, hidden)
    act_shape = config.intermediate_size if batch == 1 else (batch, config.intermediate_size)
    with _decode_graphs._no_inference_tensors():
        plan.residual = torch.empty(hidden_shape, dtype = torch.float32, device = device)
        # Unnormalized h * g: fp16 can overflow it before the consuming site applies rstd.
        plan.xg = torch.empty(hidden_shape, dtype = torch.float32, device = device)
        plan.ss = (
            torch.zeros(hidden_shape, dtype = torch.float32, device = device),
            torch.zeros(hidden_shape, dtype = torch.float32, device = device),
        )
        plan.act = torch.empty(act_shape, dtype = X.dtype, device = device)
        plan.pos = torch.zeros(batch, dtype = torch.long, device = device)
        plan.slot = torch.zeros(1, dtype = torch.long, device = device)
    plan.signature = _signature(plan)
    if batch != 1:
        geometry = tuple((e.n_heads, e.n_kv_heads, e.head_dim) for e in plan.layers)
        buffers = (plan.residual, plan.xg, *plan.ss, plan.act, plan.pos, plan.slot)
        if getattr(plan, "qkv_raw", None) is not None:
            buffers += (plan.qkv_raw,)
        shapes = tuple(tuple(t.shape) for t in buffers)
        workspaces = tuple(
            (tuple(site.H.shape), tuple(site.partials.shape), tuple(site.counter.shape))
            for entry in plan.layers
            for site in entry.sites.values()
            if site.lora
        )
        plan.signature = (plan.signature, batch, geometry, shapes, workspaces)
    _track_moves(model)
    STATS["plans"] += 1
    return plan


def decode_forward(model, X, *args):
    """The fused decode step: (normed last hidden state [B, 1, H], next cache), or None (regular path;
    nothing was written). X: the [B, 1, H] embedding of the step's token."""
    if X.device.type != "cuda":
        return None
    # Triton launches run on the current device: a model on cuda:1 while cuda:0 is current needs this.
    with torch.cuda.device(X.device):
        return _decode_forward(model, X, *args)


def _decode_forward(
    model,
    X,
    past_key_values,
    position_ids,
    attention_mask,
    rotary_seq_len,
    attention_core,
    kv_increment,
):
    if torch.is_grad_enabled() or torch.compiler.is_compiling() or not requested():
        return None
    batch = X.shape[0]
    if batch not in (1, 2, 4):
        return None
    row_shape = (-1,) if batch == 1 else (batch, -1)
    plan_key = _PLAN if batch == 1 else _BATCH_PLAN
    static = _decode_graphs.static()
    if batch != 1 and static is not None:
        return None
    if (
        batch != 1
        and model.config.model_type in _decode_graphs._SUPPORTED_MODEL_TYPES
        and _decode_graphs.enabled()
        and os.environ.get(ENV) != "force"
    ):
        return None
    # Whole-step graphs (UNSLOTH_DECODE_CUDA_GRAPHS_FULL) attend over a span with a device-side keep mask,
    # which this step does not read: let the regular path record that step.
    if static is not None and getattr(static, "span", None) is not None:
        return None
    layers = model.model.layers
    if len(layers) == 0:
        return None
    new_sequence = not hasattr(layers[0].self_attn, "paged_attention")
    plan = model.__dict__.get(plan_key)
    if (
        new_sequence
        or plan is None
        or (
            isinstance(plan, _Plan)
            and (getattr(plan, "batch_size", 1) != batch or _plan_stale(plan))
        )
    ):
        built = _build(model, X)
        if isinstance(built, str):
            STATS["refused"] = built
            built = plan if isinstance(plan, _Refused) and plan.reason == built else _Refused(built)
        if isinstance(plan, _Plan) and isinstance(built, _Plan):
            if _reusable(plan, built):
                plan.watch, plan.adapter_state = built.watch, built.adapter_state
                plan.bindings = built.bindings
                plan.tensor_bindings = built.tensor_bindings
                built = plan
        if built is not plan:
            if plan is not None:
                # Captured decode steps read the previous plan's scratch buffers.
                _decode_graphs.invalidate(model)
            plan = model.__dict__[plan_key] = built
    if isinstance(plan, _Refused) or X.device != plan.device or X.dtype != plan.dtype:
        return None
    fused_site = F.fused_site if batch == 1 else F.batch_fused_site
    final_norm = F.final_norm if batch == 1 else F.batch_final_norm
    device = X.device
    capability = plan.capability
    ropes = {}
    for entry in plan.layers:
        rope = entry.rope
        if id(rope) not in ropes:
            rope.extend_rope_embedding(X, rotary_seq_len + 1)
            cos, sin = rope.get_cached(rotary_seq_len, device.index or 0)
            if (
                cos.dim() != 2 or cos.shape != sin.shape or cos.shape[-1] != entry.head_dim
                or cos.stride(-1) != 1 or sin.stride() != cos.stride() or cos.device != device
            ):  # fmt: skip
                return None
            ropes[id(rope)] = (cos, sin)

    if position_ids.dim() == 2 and position_ids.shape[-1] > 1:
        position_ids = position_ids[:, -1:]
    if batch == 1:
        plan.pos.copy_(position_ids.reshape(-1)[-1:])
    else:
        positions = position_ids.reshape(-1)
        if positions.numel() == 1:
            positions = positions.expand(batch)
        if positions.numel() != batch:
            return None
        plan.pos.copy_(positions)
    seq_len = past_key_values[0][0].shape[-2]
    kv_seq_len = seq_len + 1
    if static is None:
        plan.slot.fill_(seq_len)
        slot = plan.slot
    else:
        slot = static.slot
    residual = plan.residual
    residual.copy_(X.view(row_shape))
    xg, ss, act, eps = plan.xg, plan.ss, plan.act, plan.eps
    n_layers = len(plan.layers)
    n_ss = 0
    next_cache = []
    for idx, entry in enumerate(plan.layers):
        attn = entry.attn
        sites = entry.sites
        n_heads, n_kv_heads, head_dim = entry.n_heads, entry.n_kv_heads, entry.head_dim
        if not hasattr(attn, "paged_attention"):
            K1, V1 = past_key_values[idx]
            _decode_graphs.alloc_decode_buffers(
                attn, K1, V1, batch, n_heads, n_kv_heads, head_dim, plan.hidden, X.dtype, device,
                kv_increment,
            )  # fmt: skip
        elif static is None and kv_seq_len >= attn.paged_attention.shape[0]:
            _decode_graphs.resize_decode_buffers(
                attn, kv_increment, batch, n_heads, n_kv_heads, head_dim
            )
        cos, sin = ropes[id(entry.rope)]
        k_cache, v_cache = attn.paged_attention_K, attn.paged_attention_V
        q = attn.temp_QA[0]
        common = dict(capability = capability, eps = eps)
        if idx == 0:
            fused_site(
                sites["qkv"], residual, q, epi = F.EPI_ROPE_KV, pre = F.PRE_INLOOP,
                norm_weight = entry.ln1, k_cache = k_cache, v_cache = v_cache,
                cache_strides = (k_cache.stride(2), k_cache.stride(0)), cos = cos, sin = sin,
                pos = plan.pos, slot = slot, head_dim = head_dim, **common,
            )  # fmt: skip
        else:
            fused_site(
                sites["qkv"], xg, q, epi = F.EPI_ROPE_KV, pre = F.PRE_SCALED, ss_in = ss[0],
                n_ss = n_ss, k_cache = k_cache, v_cache = v_cache,
                cache_strides = (k_cache.stride(2), k_cache.stride(0)), cos = cos, sin = sin,
                pos = plan.pos, slot = slot, head_dim = head_dim, **common,
            )  # fmt: skip
        Qn = q.view(batch, 1, n_heads, head_dim).transpose(1, 2)
        if static is None:
            A, Kn, Vn = attention_core(attn, Qn, kv_seq_len, attention_mask, batch, True)
            next_cache.append((Kn, Vn))
        else:
            A = static.attention(
                lambda attn = attn, Qn = Qn: attention_core(
                    attn, Qn, static.kv_len, static.mask, batch, True
                )[0]
            )
            next_cache.append((None, None))
        A = A.transpose(1, 2).reshape(row_shape)
        n1 = fused_site(
            sites["o"], A, epi = F.EPI_RESNORM, residual = residual, ss_out = ss[1],
            xg_out = xg, next_norm_weight = entry.ln2, **common,
        )  # fmt: skip
        fused_site(
            sites["gate_up"], xg, act, epi = F.EPI_SWIGLU, pre = F.PRE_SCALED, ss_in = ss[1],
            n_ss = n1, **common,
        )  # fmt: skip
        next_ln = plan.layers[idx + 1].ln1 if idx + 1 < n_layers else plan.final_norm
        n_ss = fused_site(
            sites["down"], act, epi = F.EPI_RESNORM, residual = residual, ss_out = ss[0],
            xg_out = xg, next_norm_weight = next_ln, **common,
        )  # fmt: skip
    out = torch.empty((batch, 1, plan.hidden), dtype = X.dtype, device = device)
    final_norm(residual, ss[0], n_ss, plan.final_norm, eps, out.view(row_shape))
    if static is None:
        STATS["steps"] += 1
        STATS["layers"] += n_layers
    else:
        STATS["graph_records"] += 1
    return out, next_cache
