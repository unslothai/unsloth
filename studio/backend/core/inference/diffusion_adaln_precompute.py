# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AdaLN modulation precompute: the per-block modulation Linears of FLUX.1 (norm1 / norm1_context / single norm /
norm_out) and Qwen-Image (img_mod / txt_mod / norm_out) read only the timestep (+ guidance, + pooled text)
embedding, never the image tokens. They are swapped for ``AdaLNTable`` stand-ins with no parameters; the real
modules move to host memory. Each transformer call keys its embedding inputs (timestep, guidance, pooled, ...);
the first miss of a request computes the rows of every timestep the scheduler will visit in one pass over the
modulation weights (each step at its own batch shape, so every row is the GEMM main would run: bit-identical),
then every call copies its row into one static buffer the stand-ins return.

Frees 27% (FLUX.1) / 33% (Qwen-Image, -2512, -Edit) of the transformer from the card, and those weights are never
streamed by an offloaded denoiser. Families with one shared modulation (Qwen-Image-2.1, FLUX.2, FLUX.3, Z-Image:
1-4%) are not worth the host copy and are not installed.

UNSLOTH_DIFFUSION_ADALN_PRECOMPUTE: ``0`` off, ``1`` on for every supported denoiser, ``auto`` (default) on when
the denoiser is planned offloaded / streamed / split. UNSLOTH_DIFFUSION_ADALN_VERIFY=1 checks every served row
against the module's real input (eager only; a sync per call, tests and benches)."""

from __future__ import annotations

import hashlib
import inspect
import os
import threading
import time
import weakref
from collections import OrderedDict
from typing import Any, Callable, Optional

ENV = "UNSLOTH_DIFFUSION_ADALN_PRECOMPUTE"
VERIFY_ENV = "UNSLOTH_DIFFUSION_ADALN_VERIFY"
TABLE_MB_ENV = "UNSLOTH_DIFFUSION_ADALN_TABLE_MB"
_DEFAULT_TABLE_MB = 1024
STATE_ATTR = "_unsloth_adaln"

MODE_OFF, MODE_ON, MODE_AUTO = "off", "on", "auto"

COUNTS = {
    "installs": 0,
    "served": 0,
    "rows_computed": 0,
    "weight_passes": 0,
    "precompute_s": 0.0,
    "predicted_rows": 0,
    "verify_checks": 0,
    "cross_device": 0,
}

# Input kinds: what the stand-in's module receives, from the embedding ``temb``.
_SILU = "silu"  # Linear(silu(temb))
_SILU_FIRST = (
    "silu_first"  # Linear(silu(temb.chunk(2, 0)[0])): Qwen zero_cond_t text / norm_out paths
)
_SILU_CAST = "silu_cast"  # AdaLayerNormContinuous: Linear(silu(temb).to(x.dtype))
_SILU_FIRST_CAST = "silu_first_cast"


def mode() -> str:
    raw = (os.environ.get(ENV) or "").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return MODE_OFF
    if raw in ("1", "on", "true", "yes", "force"):
        return MODE_ON
    return MODE_AUTO


def verify_enabled() -> bool:
    return (os.environ.get(VERIFY_ENV) or "").strip().lower() in ("1", "on", "true", "yes")


def _table_budget_bytes() -> int:
    try:
        mb = int(os.environ.get(TABLE_MB_ENV) or _DEFAULT_TABLE_MB)
    except ValueError:
        mb = _DEFAULT_TABLE_MB
    return max(16, mb) * 1024 * 1024


def _torch() -> Any:
    import torch
    return torch


class AdaLNTable(_torch().nn.Module):
    """Stands in for one modulation Linear; returns the armed row (a view of the transformer's static buffer). The
    real module sits in ``__dict__`` (not a submodule): never placed, streamed, cast, compiled or quantised."""

    def __init__(self, orig: Any, name: str, kind: str) -> None:
        super().__init__()
        object.__setattr__(self, "_unsloth_orig", orig)
        self.site_name = name
        self.kind = kind
        self.in_features = getattr(orig, "in_features", None)
        self.out_features = getattr(orig, "out_features", None)
        # Set per layout by the transformer's pre-hook; the same tensor object for a layout, refilled in place.
        self.unsloth_adaln_cur = None
        self.unsloth_adaln_ref = None  # verify mode: the input row the armed output came from

    def forward(self, x: Any) -> Any:
        cur = self.unsloth_adaln_cur
        if cur is None:
            raise RuntimeError(
                f"AdaLN precompute: {self.site_name} ran outside an armed transformer call"
            )
        if self.unsloth_adaln_ref is not None:
            torch = _torch()
            if not torch.compiler.is_compiling():
                COUNTS["verify_checks"] += 1
                if x.shape != self.unsloth_adaln_ref.shape or not torch.equal(
                    x, self.unsloth_adaln_ref
                ):
                    raise RuntimeError(
                        f"AdaLN precompute: {self.site_name} input differs from the precomputed row"
                    )
        if cur.device != x.device:
            # A block placed on another card (layer split, a replica): its row follows its own input.
            COUNTS["cross_device"] += 1
            return cur.to(x.device, non_blocking = True)
        return cur

    def extra_repr(self) -> str:
        return f"{self.site_name}, kind={self.kind}, in={self.in_features}, out={self.out_features}"


class _Adapter:
    cls_names: tuple = ()
    key_names: tuple = ()

    def sites(self, model: Any) -> Optional[list]:
        raise NotImplementedError

    def temb(self, model: Any, tte: Any, inputs: dict) -> Any:
        raise NotImplementedError


class _FluxAdapter(_Adapter):
    cls_names = ("FluxTransformer2DModel",)
    key_names = ("timestep", "guidance", "pooled_projections")

    def sites(self, model: Any) -> Optional[list]:
        out = []
        for i, blk in enumerate(getattr(model, "transformer_blocks", ()) or ()):
            for attr in ("norm1", "norm1_context"):
                norm = getattr(blk, attr, None)
                if norm is None or getattr(norm, "emb", None) is not None:
                    return None
                out.append((f"transformer_blocks.{i}.{attr}.linear", norm, "linear", _SILU))
        for i, blk in enumerate(getattr(model, "single_transformer_blocks", ()) or ()):
            norm = getattr(blk, "norm", None)
            if norm is None:
                return None
            out.append((f"single_transformer_blocks.{i}.norm.linear", norm, "linear", _SILU))
        norm_out = getattr(model, "norm_out", None)
        if norm_out is None or not hasattr(norm_out, "linear"):
            return None
        out.append(("norm_out.linear", norm_out, "linear", _SILU_CAST))
        return out

    def temb(self, model: Any, tte: Any, inputs: dict) -> Any:
        # FluxTransformer2DModel.forward, op for op.
        dtype = inputs["_dtype"]
        timestep = inputs["timestep"].to(dtype) * 1000
        guidance = inputs.get("guidance")
        if guidance is not None:
            guidance = guidance.to(dtype) * 1000
        pooled = inputs.get("pooled_projections")
        return tte(timestep, pooled) if guidance is None else tte(timestep, guidance, pooled)


class _QwenImageAdapter(_Adapter):
    cls_names = ("QwenImageTransformer2DModel",)
    key_names = ("timestep", "guidance", "additional_t_cond")

    def sites(self, model: Any) -> Optional[list]:
        zero = bool(getattr(model, "zero_cond_t", False))
        out = []
        for i, blk in enumerate(getattr(model, "transformer_blocks", ()) or ()):
            for attr in ("img_mod", "txt_mod"):
                seq = getattr(blk, attr, None)
                if seq is None or len(seq) != 2 or type(seq[0]).__name__ != "SiLU":
                    return None
                kind = _SILU_FIRST if (zero and attr == "txt_mod") else _SILU
                out.append((f"transformer_blocks.{i}.{attr}.1", seq, "1", kind))
        norm_out = getattr(model, "norm_out", None)
        if norm_out is None or not hasattr(norm_out, "linear"):
            return None
        out.append(
            ("norm_out.linear", norm_out, "linear", _SILU_FIRST_CAST if zero else _SILU_CAST)
        )
        return out

    def temb(self, model: Any, tte: Any, inputs: dict) -> Any:
        # QwenImageTransformer2DModel.forward, op for op (hidden_states only carries the dtype into the embedder).
        torch = _torch()
        dtype = inputs["_dtype"]
        timestep = inputs["timestep"].to(dtype)
        if getattr(model, "zero_cond_t", False):
            timestep = torch.cat([timestep, timestep * 0], dim = 0)
        guidance = inputs.get("guidance")
        if guidance is not None:
            guidance = guidance.to(dtype) * 1000
        hs = torch.empty(1, dtype = dtype, device = timestep.device)
        extra = inputs.get("additional_t_cond")
        return tte(timestep, hs, extra) if guidance is None else tte(timestep, guidance, hs, extra)


_ADAPTERS = (_FluxAdapter(), _QwenImageAdapter())


def adapter_for(model: Any) -> Optional[_Adapter]:
    name = type(model).__name__
    for ad in _ADAPTERS:
        if name in ad.cls_names:
            return ad
    return None


def _site_input(kind: str, temb: Any, dtype: Any) -> Any:
    import torch.nn.functional as F

    if kind in (_SILU_FIRST, _SILU_FIRST_CAST):
        temb = temb.chunk(2, dim = 0)[0]
    x = F.silu(temb)
    if kind in (_SILU_CAST, _SILU_FIRST_CAST):
        x = x.to(dtype)
    return x


def _pin(t: Any) -> Any:
    try:
        return t.pin_memory()
    except Exception:  # noqa: BLE001 - subclass tensors / no CUDA: pageable
        return t


def _host_pinner(modules: list) -> Callable[[Any], Any]:
    """Pins through one packed arena (diffusion_pinned_arena): a per-tensor ``pin_memory()`` rounds each weight up to
    a power of two (FLUX.1's 6.05 GiB of modulation weights would hold 7.19 GiB pinned). ``_pin`` when it is off or
    a pin fails (no CUDA: pageable)."""
    try:
        from .diffusion_pinned_arena import PinnedArena, pin_arena_enabled
        if not pin_arena_enabled():
            return _pin
        # sizes the tail slabs to what is left (the arena's ``expected_bytes``)
        arena = PinnedArena(expected_bytes = sum(_nbytes(t) for m in modules for t in _tensors(m)))
    except Exception:  # noqa: BLE001
        return _pin

    def pin(t: Any) -> Any:
        try:
            return arena.pin(t)
        except Exception:  # noqa: BLE001
            return _pin(t)

    return pin


def _tensors(mod: Any) -> list:
    return [
        t
        for m in mod.modules()
        for store in (m._parameters, m._buffers)
        for t in store.values()
        if t is not None
    ]


def _to_host(mod: Any, pin: Callable[[Any], Any] = _pin) -> int:
    """Moves ``mod``'s tensors to (pinned) host memory in place; returns bytes moved."""
    torch = _torch()
    moved = 0
    for m in mod.modules():
        for store in (m._parameters, m._buffers):
            for n, t in list(store.items()):
                if t is None:
                    continue
                moved += _nbytes(t)
                host = t.detach()
                if host.device.type != "cpu":
                    host = host.to("cpu")
                host = pin(host)
                store[n] = (
                    torch.nn.Parameter(host, requires_grad = False)
                    if store is m._parameters
                    else host
                )
    return moved


def _nbytes(t: Any) -> int:
    try:
        return int(t.numel()) * int(t.element_size())
    except Exception:  # noqa: BLE001
        return 0


def _device_clone(mod: Any, device: Any) -> Any:
    """A shallow clone of ``mod`` whose tensors live on ``device``; the host original is untouched."""
    import copy

    clone = copy.copy(mod)
    clone._parameters = {
        n: (None if t is None else t.detach().to(device, non_blocking = True))
        for n, t in mod._parameters.items()
    }
    clone._buffers = {
        n: (None if t is None else t.to(device, non_blocking = True)) for n, t in mod._buffers.items()
    }
    clone._modules = {
        n: (None if c is None else _device_clone(c, device)) for n, c in mod._modules.items()
    }
    return clone


def _key_of(tensors: list) -> tuple:
    """Exact key: dtype / shape / bytes of every embedding input (one small device-to-host copy)."""
    torch = _torch()
    meta = []
    flat = []
    for t in tensors:
        if t is None:
            meta.append(None)
            continue
        if not isinstance(t, torch.Tensor):
            meta.append(("py", repr(t)))
            continue
        meta.append((str(t.dtype), tuple(t.shape)))
        flat.append(t.detach().contiguous().reshape(-1).view(torch.uint8))
    digest = b""
    if flat:
        blob = torch.cat([f.to(flat[0].device) for f in flat]).cpu().numpy().tobytes()
        digest = hashlib.blake2b(blob, digest_size = 20).digest()
    return (tuple(meta), digest)


class _State:
    def __init__(self, model: Any, adapter: _Adapter, sites: list, tables: list, pipe: Any) -> None:
        self.model_ref = weakref.ref(model)
        self.adapter = adapter
        self.sites = sites  # [(name, parent, attr, kind)]
        self.stand_ins = tables  # AdaLNTable per site
        self.pipe_ref = None
        if pipe is not None:
            try:
                self.pipe_ref = weakref.ref(pipe)
            except TypeError:  # not weak-referenceable: hold only its scheduler lookup
                self.pipe_ref = lambda pipe = pipe: pipe
        self.lock = threading.RLock()
        self.rows: OrderedDict = OrderedDict()  # key -> (layout id, flat tensor)
        self.row_bytes = 0
        self.layouts: dict = {}  # layout id -> (flat cur buffer, [views])
        self.armed_layout = None
        self.active = True
        self.host_bytes = 0
        self.tte_host = (
            None  # host copy of the embedder used for prediction when the live one is offloaded
        )
        self.reason = ""
        self._sig = None
        self.armed_ids = None  # _input_ids of the call whose rows are armed
        self.armed_refs: tuple = ()

    def inputs(self, model: Any, args: tuple, kwargs: dict) -> dict:
        sig = self._sig
        if sig is None:
            sig = self._sig = inspect.signature(type(model).forward)
        names = [n for n in sig.parameters][1:]
        bound = dict(zip(names, args))
        bound.update(kwargs)
        hs = bound.get("hidden_states")
        out = {name: bound.get(name) for name in self.adapter.key_names}
        if out.get("timestep") is None or hs is None:
            raise _Unsupported("no timestep / hidden_states in the call")
        out["_dtype"] = hs.dtype
        out["_device"] = hs.device
        return out

    def key(self, inputs: dict) -> tuple:
        return (str(inputs["_dtype"]),) + _key_of([inputs.get(n) for n in self.adapter.key_names])

    def predicted(self, inputs: dict) -> list:
        """The inputs of every scheduler timestep this request will visit, when the call's timestep is provably
        ``t.expand(B).to(dtype) / 1000`` of one of them (FLUX / Qwen-Image pipelines); else []."""
        pipe = self.pipe_ref() if self.pipe_ref is not None else None
        sched = getattr(pipe, "scheduler", None) if pipe is not None else None
        timesteps = getattr(sched, "timesteps", None)
        ts = inputs.get("timestep")
        torch = _torch()
        if (
            timesteps is None
            or not isinstance(timesteps, torch.Tensor)
            or ts is None
            or ts.dim() != 1
        ):
            return []
        try:
            grid = timesteps.to(ts.device)
            cands = [t.expand(ts.shape[0]).to(ts.dtype) / 1000 for t in grid]
            if not cands:
                return []
            stacked = torch.stack(cands)
            hits = (stacked == ts.unsqueeze(0)).all(dim = 1).nonzero()
            if hits.numel() == 0:
                return []
            # the steps from this one on (a request never revisits an earlier one), nearest first
            cands = cands[int(hits[0, 0]) :]
        except Exception:  # noqa: BLE001 - prediction is only an optimisation
            return []
        out = []
        for c in cands:
            d = dict(inputs)
            d["timestep"] = c
            out.append(d)
        return out

    def _embedder(self, model: Any, device: Any) -> tuple:
        tte = getattr(model, "time_text_embed")
        for p in list(tte.parameters())[:1]:
            if p.device == device or (p.device.type == device.type and device.index is None):
                return tte, None
        if self.tte_host is None:
            raise _Unsupported("the time embedder is not on the call's device")
        clone = _device_clone(self.tte_host, device)
        return clone, clone

    def compute(
        self,
        model: Any,
        pending: "OrderedDict[tuple, dict]",
        protect: tuple = (),
    ) -> None:
        torch = _torch()
        t0 = time.perf_counter()
        first = next(iter(pending.values()))
        device = first["_device"]
        tte, _tmp = self._embedder(model, device)
        tembs = {}
        with torch.no_grad():
            for key, inp in pending.items():
                tembs[key] = self.adapter.temb(model, tte, inp)
            # one pass over the weights: each site onloads once, runs every pending row at its own shape
            outs: dict = {key: [] for key in pending}
            refs: dict = {key: [] for key in pending}
            keep_refs = verify_enabled()
            for (name, _parent, _attr, kind), stand in zip(self.sites, self.stand_ins):
                dev_mod = _device_clone(stand._unsloth_orig, device)
                for key, inp in pending.items():
                    x = _site_input(kind, tembs[key], inp["_dtype"])
                    y = dev_mod(x)
                    outs[key].append(y)
                    if keep_refs:
                        refs[key].append(x)
                del dev_mod
        for key in pending:
            ys = outs[key]
            outs[key] = None  # released as it is stored: one key's outputs ever held twice
            if any(y.dtype != ys[0].dtype or y.device != ys[0].device for y in ys):
                raise _Unsupported("modulation outputs mix dtypes / devices")
            layout_id = tuple((tuple(y.shape), str(y.dtype), str(y.device)) for y in ys)
            flat = torch.cat([y.reshape(-1) for y in ys])
            del ys
            self._store(key, layout_id, flat, refs[key] if keep_refs else None, protect = protect)
            del flat
            COUNTS["rows_computed"] += 1
        COUNTS["weight_passes"] += 1
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        COUNTS["precompute_s"] += time.perf_counter() - t0

    def _store(
        self,
        key: tuple,
        layout_id: tuple,
        flat: Any,
        refs: Optional[list],
        protect: tuple = (),
    ) -> None:
        old = self.rows.pop(key, None)
        if old is not None:
            self.row_bytes -= _nbytes(old[1])
        self.rows[key] = (layout_id, flat, refs)
        self.row_bytes += _nbytes(flat)
        self._evict(protect)

    def row_bytes_estimate(self, model: Any, inputs: dict) -> int:
        """Bytes of one key's row (every site's output at this call's batch), before computing it."""
        torch = _torch()
        batch = int(inputs["timestep"].shape[0]) if inputs["timestep"].dim() else 1
        doubled = bool(getattr(model, "zero_cond_t", False))
        size = torch.empty((), dtype = inputs["_dtype"]).element_size()
        total = 0
        for (_name, _parent, _attr, kind), stand in zip(self.sites, self.stand_ins):
            rows = 2 * batch if (doubled and kind == _SILU) else batch
            total += rows * int(stand.out_features or 0) * size
        return max(1, total)

    def make_room(self, model: Any, pending: "OrderedDict[tuple, dict]", protect: tuple) -> None:
        """Bounds what a miss computes by the table budget BEFORE any output is allocated: predictions beyond it are
        dropped (computed lazily when their step comes), and older rows are evicted to make room for the rest."""
        budget = _table_budget_bytes()
        est = self.row_bytes_estimate(model, next(iter(pending.values())))
        fit = max(len(protect), budget // est)
        for key in list(pending)[::-1]:
            if len(pending) <= fit:
                break
            if key not in protect:
                del pending[key]
        self._evict(protect, budget = max(0, budget - len(pending) * est))

    def _evict(
        self,
        protect: tuple = (),
        budget: Optional[int] = None,
    ) -> None:
        """Oldest first, never a ``protect`` row (the call being served): over budget, predictions go first."""
        budget = _table_budget_bytes() if budget is None else budget
        if self.row_bytes <= budget:
            return
        for k in list(self.rows):
            if self.row_bytes <= budget:
                break
            if k in protect:
                continue
            _l, f, _r = self.rows.pop(k)
            self.row_bytes -= _nbytes(f)

    def _layout(self, layout_id: tuple, like: Any) -> tuple:
        lay = self.layouts.get(layout_id)
        if lay is None:
            torch = _torch()
            total = sum(int(torch.Size(s).numel()) for s, _d, _v in layout_id)
            # A normal tensor even when the first miss runs under inference_mode: it is refilled in place by every
            # later call, inference_mode or not (an inference tensor refuses that outside it).
            with torch.inference_mode(False):
                buf = torch.empty(total, dtype = like.dtype, device = like.device)
            views = []
            off = 0
            for shape, _d, _v in layout_id:
                n = int(torch.Size(shape).numel())
                views.append(buf[off : off + n].view(shape))
                off += n
            lay = self.layouts[layout_id] = (buf, views)
        return lay

    def arm(self, key: tuple) -> None:
        layout_id, flat, refs = self.rows[key]
        self.rows.move_to_end(key)
        buf, views = self._layout(layout_id, flat)
        buf.copy_(flat, non_blocking = True)
        if self.armed_layout is not layout_id:
            for stand, view in zip(self.stand_ins, views):
                stand.unsloth_adaln_cur = view
            self.armed_layout = layout_id
        for i, stand in enumerate(self.stand_ins):
            stand.unsloth_adaln_ref = refs[i] if refs is not None else None
        COUNTS["served"] += 1


class _Unsupported(Exception):
    pass


def _pre_hook(model: Any, args: tuple, kwargs: dict) -> None:
    st = model.__dict__.get(STATE_ATTR)
    if st is None or not st.active:
        return None
    try:
        _serve(model, st, args, kwargs)
    except _Unsupported as exc:
        # Cannot key this call: put the real modules back on the call's device and run stock from here on.
        device = None
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is not None and hasattr(hs, "device"):
            device = hs.device
        uninstall(model, device = device, reason = f"unsupported call: {exc}")
    return None


def replica_memo(module: Any, device: Any, memo: dict) -> None:
    """CFG-replica policy (diffusion_multigpu._copy_module): the replica takes the REAL modulation modules on its own
    card and no precompute state. The source's pre-hook arms only the source's stand-ins, so a copied stand-in would
    keep stale rows; the state (lock, tables on the source card) is source-only and never traversed."""
    import copy

    st = module.__dict__.get(STATE_ATTR)
    if st is None:
        return
    for stand in st.stand_ins:
        memo[id(stand)] = copy.deepcopy(stand._unsloth_orig).to(device)


def _input_ids(st: "_State", inputs: dict) -> Optional[tuple]:
    """Identity of a call's embedding inputs (objects, storage, in-place version): equal ids, equal rows. None when
    an input cannot report mutation (an inference tensor, Studio's renders run under inference_mode, carries no
    version counter): such calls always re-arm."""
    out = [str(inputs.get("_dtype"))]
    for name in st.adapter.key_names:
        t = inputs.get(name)
        is_inference = getattr(t, "is_inference", None)
        if t is not None and (not callable(is_inference) or is_inference()):
            return None
        out.append(
            None
            if t is None
            else (
                id(t),
                getattr(t, "data_ptr", lambda: 0)(),
                getattr(t, "_version", None),
                tuple(t.shape),
            )
        )
    return tuple(out)


def _arm_call(model: Any, args: tuple, kwargs: dict) -> None:
    st = model.__dict__.get(STATE_ATTR)
    if st is None or not st.active:
        return
    try:
        if (
            st.armed_ids is not None
            and _input_ids(st, st.inputs(model, args, kwargs)) == st.armed_ids
        ):
            return  # the rows armed last are this call's own: no key read, no copy
    except _Unsupported:
        pass
    _pre_hook(model, args, kwargs)


def arm_call(model: Any, args: tuple, kwargs: dict) -> None:
    """Arms the rows for a call that reaches the forward without ``model.__call__`` (the CFG router's merged
    cond / uncond batch, its plain fallbacks and timing calls): its own inputs, so every call reads its own rows."""
    _torch().compiler.disable(_arm_call)(model, args, kwargs)


def offload_reason(model: Any) -> Optional[str]:
    """Why ``model`` streams (offload machinery holds per-module bookkeeping a restored module would miss), or None."""
    for name, sub in model.named_modules():
        if getattr(sub, "_hf_hook", None) is not None:
            return f"{name or '<root>'} carries an accelerate hook"
        registry = sub.__dict__.get("_diffusers_hook")
        hooks = getattr(registry, "hooks", None) or {}
        if any("offload" in str(k).lower() or "prefetch" in str(k).lower() for k in hooks):
            return f"{name or '<root>'} is group offloaded"
        if "_unsloth_group_prefetcher" in sub.__dict__:
            return f"{name or '<root>'} streams through the group prefetcher"
    return None


def _serve(model: Any, st: "_State", args: tuple, kwargs: dict) -> None:
    with st.lock:
        inputs = st.inputs(model, args, kwargs)
        key = st.key(inputs)
        if key not in st.rows:
            pending: OrderedDict = OrderedDict()
            pending[key] = inputs
            for pred in st.predicted(inputs):
                pk = st.key(pred)
                if pk not in st.rows and pk not in pending:
                    pending[pk] = pred
                    COUNTS["predicted_rows"] += 1
            st.make_room(model, pending, (key,))
            st.compute(model, pending, protect = (key,))
        st.arm(key)
        st.armed_ids = _input_ids(st, inputs)
        # held while armed: a freed input's id / storage could otherwise be reused by the next step's tensor
        st.armed_refs = tuple(inputs.get(name) for name in st.adapter.key_names)


def decline_reason(
    model: Any,
    *,
    offloaded: bool,
    requested: Optional[str] = None,
    split: bool = False,
) -> Optional[str]:
    """None when ``model`` should take the lever under the current mode."""
    m = requested or mode()
    if m == MODE_OFF:
        return f"{ENV}=0"
    if model is None:
        return "no transformer"
    if split:
        # Blocks on several cards: one static row buffer cannot sit beside all of them.
        return "the denoiser is split across devices"
    if adapter_for(model) is None:
        return f"{type(model).__name__} has no per-block modulation adapter"
    if m == MODE_AUTO and not offloaded:
        return "auto: the denoiser is resident (set UNSLOTH_DIFFUSION_ADALN_PRECOMPUTE=1 to force)"
    if getattr(model, "peft_config", None):
        return "LoRA adapters are attached"
    return None


def _pipe_loras(pipe: Any) -> bool:
    return bool(
        getattr(pipe, "_unsloth_loras", None) or getattr(pipe, "_unsloth_loras_baked", False)
    )


def _small_host(pipe: Any) -> bool:
    from .diffusion_small_host import SMALL_HOST_ATTR
    return bool(getattr(pipe, SMALL_HOST_ATTR, None))


def _hook_bound(module: Any) -> bool:
    """A module (or submodule) whose ``forward`` is an instance attribute: diffusers hooks (layerwise casting, group
    offload) and Accelerate hooks bind it to that module, so a ``_device_clone`` would still run on the original's
    host weights."""
    return any(
        "forward" in sub.__dict__ or "_diffusers_hook" in sub.__dict__ or "_hf_hook" in sub.__dict__
        for sub in module.modules()
    )


def installed(model: Any) -> bool:
    st = getattr(model, "__dict__", {}).get(STATE_ATTR)
    return st is not None and st.active


def install(
    model: Any,
    pipe: Any = None,
    *,
    offloaded: bool = False,
    logger: Any = None,
    requested: Optional[str] = None,
    split: bool = False,
) -> bool:
    """Swaps the modulation Linears of a supported denoiser for ``AdaLNTable`` stand-ins (weights to host)."""
    reason = decline_reason(model, offloaded = offloaded, requested = requested, split = split)
    if reason is None and pipe is not None and _pipe_loras(pipe):
        # baked into the weights before quantisation and re-merged in place later: the modules must stay in the tree
        reason = "LoRA adapters are baked into this load"
    if reason is None and pipe is not None and _small_host(pipe):
        # the route keeps the denoiser memory-mapped or int8 in pageable / file-backed memory on a host short of RAM:
        # pinning the modulation weights would add page-locked RAM the route chose not to take
        reason = "the small-host route keeps the denoiser memory-mapped or int8 off pinned memory"
    if reason is not None:
        if logger is not None:
            logger.info("diffusion.adaln_precompute: off (%s)", reason)
        return False
    if installed(model):
        return True
    adapter = adapter_for(model)
    sites = adapter.sites(model)
    if not sites or not hasattr(model, "time_text_embed"):
        if logger is not None:
            logger.info(
                "diffusion.adaln_precompute: off (unrecognised %s layout)", type(model).__name__
            )
        return False
    import copy

    # A layer whose output depends on the step beyond its input (NVFP4 per-step protection runs protected steps at
    # another precision) stays live: a precomputed row would ignore the step's mode. So does a hook-bound layer
    # (_hook_bound): its device clone would compute with the host original.
    def _stays_live(mod: Any) -> bool:
        return _hook_bound(mod) or any(
            getattr(m, "protect", None) is not None for m in mod.modules()
        )

    live = [
        name
        for name, parent, attr, _kind in sites
        if _stays_live(parent._modules.get(attr) or getattr(parent, attr))
    ]
    if live:
        sites = [site for site in sites if site[0] not in set(live)]
        if logger is not None:
            logger.info(
                "diffusion.adaln_precompute: %d step-protected modulation layer(s) stay live",
                len(live),
            )
        if not sites:
            return False
    stands = []
    moved = 0
    pin = _host_pinner(
        [
            parent._modules[attr] if attr in parent._modules else getattr(parent, attr)
            for _n, parent, attr, _k in sites
        ]
    )
    for name, parent, attr, kind in sites:
        orig = parent._modules[attr] if attr in parent._modules else getattr(parent, attr)
        moved += _to_host(orig, pin)
        stand = AdaLNTable(orig, name, kind)
        parent._modules[attr] = stand
        stands.append(stand)
    st = _State(model, adapter, sites, stands, pipe)
    st.host_bytes = moved
    try:
        st.tte_host = copy.deepcopy(model.time_text_embed).to("cpu")
    except Exception:  # noqa: BLE001
        st.tte_host = None
    model.__dict__[STATE_ATTR] = st
    # Always eager: a whole-module compile graph-breaks here instead of tracing the keyed lookup.
    torch = _torch()
    st.hook = model.register_forward_pre_hook(torch.compiler.disable(_pre_hook), with_kwargs = True)
    COUNTS["installs"] += 1
    try:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    if logger is not None:
        logger.info(
            "diffusion.adaln_precompute: on (%s, %d modulation sites, %.2f GiB moved to host)",
            type(model).__name__,
            len(stands),
            moved / 2**30,
        )
    return True


def _owner_device(model: Any, site_name: str) -> Any:
    """Device of the nearest enclosing module that still holds weights (the stand-in's parent norm or modulation
    Sequential holds none once the Linear is swapped out): the block, else the transformer itself."""
    parts = site_name.split(".")[:-1]
    while True:
        prefix = ".".join(parts)
        try:
            owner = model.get_submodule(prefix) if prefix else model
        except AttributeError:
            owner = None
        if owner is not None:
            for p in owner.parameters():
                return p.device
        if not parts:
            return None
        parts.pop()


def _group_host_copy(go: Any, t: Any, low_cpu_mem_usage: bool) -> Any:
    """``ModuleGroup._to_cpu`` where diffusers has it; older diffusers (0.36, the Python < 3.10 pin) build the same copy
    inline in ``_init_cpu_param_dict``."""
    to_cpu = getattr(go.ModuleGroup, "_to_cpu", None)
    if to_cpu is not None:
        return to_cpu(t, low_cpu_mem_usage)
    host = t.data.cpu()
    return host if low_cpu_mem_usage else host.pin_memory()


def _rebuild_streamed(model: Any, restored: list) -> set:
    """Registers restored modules with the offload machinery that streams their block, as if they had been there
    when it was built: each param / buffer gets its host copy in the owning diffusers group's ``cpu_param_dict``
    (made by diffusers' own ``ModuleGroup._to_cpu``), lands on the onload device if its group is on it now
    (resident groups), and every Studio ``GroupPrefetcher`` re-sizes that group (bytes, window, slot views; the
    slot ring is rebuilt when the group outgrew it). Returns the ids of the modules a group now owns."""
    from .diffusion_memory import _offload_groups

    groups = _offload_groups(model)
    if not groups:
        return set()
    from diffusers.hooks import group_offloading as go

    from .diffusion_offload_prefetch import (
        PREFETCHER_ATTR,
        _group_nbytes,
        _group_tensors,
        _packed_bytes,
        bump_placement_epoch,
    )

    owner: dict = {}
    for group in groups:
        for module in getattr(group, "modules", None) or ():
            for sub in module.modules():
                owner.setdefault(id(sub), group)
    owned: set = set()
    touched: dict = {}
    for orig in restored:
        group = owner.get(id(orig))
        if group is None:
            continue
        if getattr(group, "offload_to_disk_path", None):
            raise _Unsupported("disk-offloaded groups key their tensors by file")
        mine = {id(t) for t in list(orig.parameters()) + list(orig.buffers())}
        others = [t for t in _group_tensors(group) if id(t) not in mine]
        onloaded = any(t.device.type != "cpu" for t in others)
        for t in list(orig.parameters()) + list(orig.buffers()):
            if getattr(group, "stream", None) is not None:
                group.cpu_param_dict[t] = _group_host_copy(
                    go, t, bool(getattr(group, "low_cpu_mem_usage", False))
                )
            if onloaded:
                t.data = t.data.to(group.onload_device)
        owned.add(id(orig))
        touched[id(group)] = group
    if not touched:
        return owned
    for sub in model.modules():
        pf = sub.__dict__.get(PREFETCHER_ATTR)
        if pf is None:
            continue
        grew = False
        for gid, group in touched.items():
            if gid not in getattr(pf, "by_id", {}):
                continue
            pf.nbytes[gid] = _group_nbytes(group)
            pf.slot_buffers.pop(gid, None)
            if pf.slot_of and gid in pf.slot_of:
                cpu = getattr(group, "cpu_param_dict", None) or {}
                need = _packed_bytes([cpu.get(t, t) for t in _group_tensors(group)])
                grew = grew or need is None or need > pf.slot_size
        largest = max(pf.nbytes.values()) if pf.nbytes else 0
        pf.window = max(int(pf.window), (pf.depth + 1) * largest)
        if grew:
            pf.disable_slots()
            pf.enable_slots()
    bump_placement_epoch()
    return owned


def _sequential_offload_hook(model: Any) -> Any:
    """An Accelerate AlignDevicesHook that offloads its module's weights (enable_sequential_cpu_offload), or None."""
    for sub in model.modules():
        hook = getattr(sub, "_hf_hook", None)
        for h in getattr(hook, "hooks", None) or ((hook,) if hook is not None else ()):
            if type(h).__name__ == "AlignDevicesHook" and getattr(h, "offload", False):
                return h
    return None


def uninstall(
    model: Any,
    *,
    device: Any = None,
    logger: Any = None,
    reason: str = "",
) -> bool:
    """Puts the real modulation modules back: under offload machinery they are registered with the group that
    streams their block (``_rebuild_streamed``); elsewhere on ``device``, else beside their block's weights."""
    st = getattr(model, "__dict__", {}).get(STATE_ATTR)
    if st is None:
        return False
    torch = _torch()
    with st.lock:
        st.active = False
        try:
            st.hook.remove()
        except Exception:  # noqa: BLE001
            pass
        # targets first: once a module is back, its own host params would answer the owner lookup
        targets = [
            device if device is not None else _owner_device(model, name)
            for name, *_rest in st.sites
        ]
        restored = []
        for (name, parent, attr, _kind), stand in zip(st.sites, st.stand_ins):
            orig = stand._unsloth_orig
            parent._modules[attr] = orig
            restored.append(orig)
        owned = _rebuild_streamed(model, restored)
        hook = _sequential_offload_hook(model)
        for orig, target in zip(restored, targets):
            if id(orig) in owned:
                continue
            if hook is not None:
                # sequential CPU offload: hook it the way enable_sequential_cpu_offload hooked its siblings (its
                # weights move into the hook's weights map, the params to meta, onloaded per forward)
                from accelerate import cpu_offload
                cpu_offload(
                    orig,
                    execution_device = hook.execution_device,
                    offload_buffers = bool(getattr(hook, "offload_buffers", False)),
                )
                continue
            if target is not None and str(target) not in ("cpu", "meta"):
                orig.to(target)
        st.rows.clear()
        st.layouts.clear()
    del model.__dict__[STATE_ATTR]
    if logger is not None:
        logger.info("diffusion.adaln_precompute: removed (%s)", reason or "requested")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def bind_pipe(pipe: Any) -> None:
    """Points the prediction at the scheduler of the pipe about to render (derived pipes share the denoiser)."""
    model = getattr(pipe, "transformer", None)
    st = getattr(model, "__dict__", {}).get(STATE_ATTR) if model is not None else None
    if st is None:
        return
    try:
        st.pipe_ref = weakref.ref(pipe)
    except TypeError:
        st.pipe_ref = lambda pipe = pipe: pipe


def summary(model: Any) -> dict:
    st = getattr(model, "__dict__", {}).get(STATE_ATTR)
    if st is None:
        return {"active": False}
    return {
        "active": st.active,
        "sites": len(st.stand_ins),
        "host_gib": round(st.host_bytes / 2**30, 3),
        "cached_rows": len(st.rows),
        "table_mib": round(st.row_bytes / 2**20, 1),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in COUNTS.items()},
    }
