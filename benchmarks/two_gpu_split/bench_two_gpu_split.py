#!/usr/bin/env python3
"""Two-GPU layer split for a diffusers transformer: does it work here, and what does it cost?

Self-contained. Needs torch + diffusers (+ accelerate for the dispatch_model row, + flashinfer
only if your own quantised layers use it). Nothing else is imported, nothing is written outside
--out, no torch.distributed, no NCCL, no os.fork, no signals, no Unix-only calls. Runs on
Windows, WSL and Linux, on any two CUDA devices, with or without peer-to-peer access.

WHAT IT DOES
------------
Splits the repeated transformer blocks of a diffusers DiT across two GPUs -- blocks 0..k-1 on the
first, k..n-1 on the second -- with plain per-module .to(device) and an explicit copy of the
tensors that cross the boundary. Everything that is not a repeated block (patch/time/text
embedders, rotary tables, the modulation tail, norm_out, proj_out) stays on the first card,
because those are reached from code paths that build helper tensors from x.device and have no
module boundary to intercept. Then it renders the same seed on one GPU and on two and compares
the images pixel for pixel, and times the split against every alternative you have today.

  python bench_two_gpu_split.py --repo Tongyi-MAI/Z-Image-Turbo --steps 9 --size 1024 \
      --devices 0,1 --reps 5 --out split_report.json

  # your own two 5090s, a bigger model, and the compiled deployment mode:
  python bench_two_gpu_split.py --repo <repo-or-local-path> --compile --reps 7

HOW TO READ THE OUTPUT
----------------------
  P2P                 whether the driver can copy card to card directly. False is normal for
                      GeForce over PCIe and is NOT a failure: the driver stages the copy through
                      its own host buffer. The host-staged row is measured either way, so the two
                      paths are separable on a machine that has P2P.
  bit-identical       max |single-GPU image - split image|. 0 means the split changed nothing.
                      Anything else is reported with the pixel count, not hidden behind a
                      tolerance. Two cards of DIFFERENT models (a 5090 plus a Spark) are not
                      expected to be bit-identical: different SM counts change the tile/split-k
                      choice inside cuBLAS and therefore the reduction order.
  boundary            bytes and milliseconds per forward that cross the PCIe/NVLink boundary.
                      Compare ms/forward against s/render divided by steps: that ratio is what
                      the split costs you.
  compile             reported in its own column, never inside s/render. Moving a block to
                      another card makes torch.compile recompile it (the device is in the
                      guards), so a split has a real one-off compile cost that a single GPU
                      does not.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
from torch import nn

__version__ = "1.0"


# ===================================================================== byte + device accounting
def leaf_storages(t, out):
    """(storage key, nbytes) for every real storage under t, unwrapping tensor subclasses.

    A packed 4-bit weight has no dtype whose itemsize is 0.5 and a torchao quantised weight is a
    subclass whose outer storage is invalid, so bytes are counted through untyped storages. Both
    matter for a memory-balanced split: the point is to balance the bytes actually resident.
    """
    if type(t) is not torch.Tensor and hasattr(t, "__tensor_flatten__"):
        try:
            names, _ = t.__tensor_flatten__()
        except Exception:
            names = []
        if names:
            for n in names:
                inner = getattr(t, n, None)
                if inner is not None:
                    leaf_storages(inner, out)
            return
    try:
        st = t.untyped_storage()
        out.append(((st.data_ptr(), st.nbytes()), st.nbytes()))
    except Exception:
        out.append(((id(t), t.numel()), t.numel() * t.element_size()))


def module_bytes(module, seen = None):
    own = seen if seen is not None else set()
    total = 0
    for t in list(module.parameters(recurse = True)) + list(module.buffers(recurse = True)):
        if t is None:
            continue
        found = []
        leaf_storages(t, found)
        for key, nbytes in found:
            if key in own:
                continue
            own.add(key)
            total += nbytes
    return total


def module_device(module):
    for t in list(module.parameters(recurse = True)) + list(module.buffers(recurse = True)):
        if t is not None and t.device.type == "cuda":
            return t.device
    for t in list(module.parameters(recurse = True)) + list(module.buffers(recurse = True)):
        if t is not None:
            return t.device
    return None


def free_all():
    gc.collect()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()


def sync_all():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


# ================================================================================= the boundary
class BoundaryTransfer:
    """One device crossing: copy every tensor argument into persistent buffers on the target.

    Persistent buffers rather than src.to(device) for three reasons: a CUDA graph replay needs
    static addresses on both sides; the destination allocator is not asked for a fresh block on
    every block of every step; and the copy becomes a pure memcpy whose time can be attributed
    with two CUDA events instead of being tangled with an allocation.

    stage="direct"  dst.copy_(src). A peer memcpy where P2P exists; where it does not, the driver
                    stages it through its own host buffer, which is what a pair of GeForce cards
                    over PCIe actually does.
    stage="host"    src -> pinned CPU buffer -> dst, explicitly. Measured on both kinds of
                    machine so the no-P2P cost is a number rather than a guess.
    """

    def __init__(
        self,
        name,
        src,
        dst,
        stage = "direct",
        instrument = False,
    ):
        self.name, self.src, self.dst, self.stage = name, src, dst, stage
        self.instrument = instrument
        self.bufs, self.host_bufs, self.memo = {}, {}, {}
        self.n_calls = self.n_leader = self.n_bytes = self.n_memo_hits = 0
        self.ms = 0.0
        self.shapes = {}
        self._pending = []

    def _buf(self, key, ref):
        b = self.bufs.get(key)
        if b is None:
            with torch.inference_mode(False):
                b = torch.empty(ref.shape, dtype = ref.dtype, device = self.dst)
            self.bufs[key] = b
        return b

    def _host(self, key, ref):
        b = self.host_bufs.get(key)
        if b is None:
            with torch.inference_mode(False):
                b = torch.empty(ref.shape, dtype = ref.dtype, device = "cpu", pin_memory = True)
            self.host_bufs[key] = b
        return b

    def move(
        self,
        obj,
        kp = (),
    ):
        if torch.is_tensor(obj):
            if obj.device == self.dst:
                return obj
            hit = self.memo.get(id(obj))
            if hit is not None and hit[0] is obj:
                self.n_memo_hits += 1
                return hit[1]
            key = (kp, tuple(obj.shape), obj.dtype)
            dst = self._buf(key, obj)
            nb = obj.numel() * obj.element_size()
            self.n_bytes += nb
            self.shapes[str(kp)] = [list(obj.shape), str(obj.dtype), nb]
            if self.stage == "host":
                h = self._host(key, obj)
                h.copy_(obj, non_blocking = False)
                dst.copy_(h, non_blocking = False)
            else:
                dst.copy_(obj, non_blocking = True)
            self.memo[id(obj)] = (obj, dst)
            return dst
        if isinstance(obj, (list, tuple)):
            kind = type(obj)
            vals = [self.move(o, kp + (i,)) for i, o in enumerate(obj)]
            try:
                return kind(vals)
            except TypeError:  # namedtuple and friends
                return tuple(vals)
        if isinstance(obj, dict):
            return {k: self.move(v, kp + (k,)) for k, v in obj.items()}
        return obj

    def _set_device(self):
        """Make the destination card the CURRENT CUDA device for the blocks that follow.

        Not redundant with moving the tensors. ATen operators get a device guard from the
        dispatcher, which is why bf16 and fp8 (torch._scaled_mm) split correctly on tensor
        placement alone. A torch.library custom op whose body calls a third-party extension does
        NOT: flashinfer's cutlass FP4 GEMM takes its stream from the tensor
        (get_stream(mat1.device())) and installs no CUDADeviceGuard, so with the current device
        still card 0 the launch goes into card 0's context holding card 1's pointers. That is an
        illegal memory access, and on one machine here it left the card reporting
        "GPU requires reset". If you run any custom quantised kernel, you need this.

        torch.compiler.is_compiling() is a compile-time constant, so under torch.compile the
        branch folds away and adds no graph break; inductor's generated wrapper already opens a
        device guard for the graph's device.
        """
        if self.dst.type != "cuda" or torch.compiler.is_compiling():
            return
        if torch.cuda.current_device() != self.dst.index:
            torch.cuda.set_device(self.dst)

    def __call__(
        self,
        args,
        kwargs,
        leader = False,
    ):
        self.n_calls += 1
        self._set_device()
        if leader:
            self.n_leader += 1
            # Only the hidden state is threaded through the block loop; the encoder hidden
            # states, the rotary table and the modulation vector are loop invariants handed to
            # every block from the first card. The leader copies them once per forward and the
            # rest of the run looks them up by object identity, which is why the memo is cleared
            # here and not on every call: a modulation vector that changes per denoising step
            # must never be served stale.
            self.memo.clear()
        if not (self.instrument and leader):
            return self.move(args, ("a",)), self.move(kwargs, ("k",))
        # Both events on the SOURCE stream: a cross-device elapsed_time is undefined and raises.
        s = torch.cuda.current_stream(self.src)
        e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        e0.record(s)
        out = (self.move(args, ("a",)), self.move(kwargs, ("k",)))
        e1.record(s)
        self._pending.append((e0, e1))
        return out

    def drain(self):
        for e0, e1 in self._pending:
            try:
                e1.synchronize()
                self.ms += e0.elapsed_time(e1)
            except Exception:
                pass
        self._pending = []

    def stats(self):
        self.drain()
        n = max(self.n_leader or self.n_calls, 1)
        return {
            "name": self.name,
            "src": str(self.src),
            "dst": str(self.dst),
            "stage": self.stage,
            "calls": self.n_calls,
            "forwards": self.n_leader,
            "memo_hits": self.n_memo_hits,
            "bytes_per_forward": self.n_bytes / n,
            "bytes_total": self.n_bytes,
            "ms_per_forward": self.ms / n,
            "instrumented": self.instrument,
            "tensors": self.shapes,
            "persistent_buffer_bytes": sum(
                b.numel() * b.element_size() for b in self.bufs.values()
            ),
        }

    def release(self):
        self.bufs, self.host_bufs, self.memo, self._pending = {}, {}, {}, []


class BoundaryShim(nn.Module):
    """Performs the boundary copy OUTSIDE the compiled block.

    compile_repeated_blocks matches submodules by CLASS NAME, so this wrapper is never compiled
    while the block it wraps still is. That matters: a forward pre-hook is traced INTO the
    compiled region (Module.compile wraps _call_impl, which is what dispatches hooks), and a
    CUDA event recorded there is rejected by torch because inductor applies a compiled region's
    input mutations in a copy_ epilogue after the graph body.
    """

    def __init__(self, block, transfer, leader):
        super().__init__()
        self.block = block
        self._transfer = transfer
        self._leader = leader

    def forward(self, *args, **kwargs):
        args, kwargs = self._transfer(args, kwargs, self._leader)
        return self.block(*args, **kwargs)


# ==================================================================================== placement
class Split:
    """State of one placement; keep it and call undo()."""

    def __init__(self):
        self.handles, self.shims, self.transfers = [], [], []
        self.assignment, self.reasoning, self.placement = {}, {}, {}
        self.boundary = "hook"

    def transfer_stats(self):
        return [t.stats() for t in self.transfers]

    def undo(self, model, primary):
        for mlist, idx, original in self.shims:
            mlist[idx] = original
        for h in self.handles:
            h.remove()
        for t in self.transfers:
            t.release()
        self.handles, self.shims, self.transfers = [], [], []
        model.to(primary)
        free_all()


def find_containers(model):
    """Every nn.ModuleList whose children are all repeated blocks, in declaration order."""
    classes = tuple(
        getattr(model, "_repeated_blocks", None) or getattr(model, "_no_split_modules", None) or ()
    )
    if not classes:
        raise SystemExit(
            f"{type(model).__name__} declares neither _repeated_blocks nor "
            "_no_split_modules; this script cannot tell which modules repeat"
        )
    out = []
    for fqn, mod in model.named_modules():
        if (
            isinstance(mod, nn.ModuleList)
            and len(mod)
            and all(c.__class__.__name__ in classes for c in mod)
        ):
            out.append((fqn, mod))
    if not out:
        raise SystemExit(f"no nn.ModuleList of {classes} found in {type(model).__name__}")
    return classes, out


def census(model):
    classes, containers = find_containers(model)
    seen = set()
    block_bytes = {fqn: [module_bytes(b, seen) for b in ml] for fqn, ml in containers}
    total_seen = set()
    total = module_bytes(model, total_seen)
    block_total = sum(sum(v) for v in block_bytes.values())
    return {
        "class": type(model).__name__,
        "block_classes": list(classes),
        "containers": [
            {"fqn": f, "n": len(ml), "bytes": sum(block_bytes[f])} for f, ml in containers
        ],
        "block_bytes": block_bytes,
        "containers_raw": containers,
        "non_block_bytes": total - block_total,
        "total_bytes": total,
    }


def plan_cut(
    block_bytes,
    fixed_a,
    ratio = None,
):
    """Memory-balanced cut, or the user's own block fraction."""
    n = len(block_bytes)
    if ratio is not None:
        k = max(0, min(n, int(round(ratio * n))))
        return k, {"policy": "ratio", "ratio": ratio}
    pre = [0]
    for b in block_bytes:
        pre.append(pre[-1] + b)
    tot = pre[-1]
    best, best_k = None, 0
    for k in range(n + 1):
        imb = abs((fixed_a + pre[k]) - (tot - pre[k]))
        if best is None or imb < best:
            best, best_k = imb, k
    return best_k, {"policy": "balanced", "imbalance_bytes": best}


def split_transformer(
    model,
    devices,
    *,
    ratio = None,
    stage = "direct",
    boundary = "hook",
    instrument = False,
):
    devs = [torch.device(d) for d in devices]
    primary = devs[0]
    c = census(model)
    containers = c["containers_raw"]
    target_fqn, target = max(containers, key = lambda t: sum(c["block_bytes"][t[0]]))

    sp = Split()
    sp.boundary = boundary
    # Everything that is not the splittable stack goes to the first card, unconditionally, so a
    # re-split at a different ratio starts from a known state.
    for name, child in model.named_children():
        if name != target_fqn:
            child.to(primary)
    for p in model.parameters(recurse = False):
        p.data = p.data.to(primary)
    for b in model.buffers(recurse = False):
        b.data = b.data.to(primary)

    fixed_a = c["non_block_bytes"] + sum(
        sum(c["block_bytes"][f]) for f, _ in containers if f != target_fqn
    )
    bb = c["block_bytes"][target_fqn]
    k, why = plan_cut(bb, fixed_a, ratio)
    assign = [0] * k + [1] * (len(bb) - k)
    sp.assignment[target_fqn] = assign
    why["cut"] = k
    sp.reasoning[target_fqn] = why
    for i, blk in enumerate(target):
        want = devs[assign[i]]
        if module_device(blk) != want:
            blk.to(want)

    # A block needs the boundary applied if its predecessor left the hidden state on another
    # card OR it is itself not on the first card and therefore cannot read the loop invariants.
    run_tr = None
    for i in range(len(assign)):
        want = devs[assign[i]]
        prev = primary if i == 0 else devs[assign[i - 1]]
        starts = want != prev
        if not (starts or want != primary):
            run_tr = None
            continue
        leader = starts or run_tr is None
        if leader:
            run_tr = BoundaryTransfer(
                f"{target_fqn}.{i}:{prev}->{want}",
                prev if starts else primary,
                want,
                stage,
                instrument,
            )
            sp.transfers.append(run_tr)
        _install(sp, target, i, run_tr, boundary, leader)
    last = devs[assign[-1]]
    if last != primary:
        tr = BoundaryTransfer(
            f"{target_fqn}.out:{last}->{primary}", last, primary, stage, instrument
        )
        sp.transfers.append(tr)
        sp.handles.append(
            target[len(assign) - 1].register_forward_hook(
                lambda m, a, out, _t = tr: _t((out,), {}, True)[0][0]
            )
        )

    # Keep the process pointed at the first card. The boundary makes the far card current so
    # that kernels without their own device guard (flashinfer's FP4 GEMM is one) launch into the
    # right context, and something has to put it back: any index-free allocation in the library
    # (torch.zeros(..., device="cuda"), an arange in a timestep embedder) follows the CURRENT
    # device, so leaving it on the far card puts helper tensors on the wrong GPU and the failure
    # appears in a module that never moved.
    p0 = torch.device(primary)
    if p0.type == "cuda":
        if torch.cuda.current_device() != p0.index:
            torch.cuda.set_device(p0)

        def _primary_guard(_m, _a, _kw):
            if not torch.compiler.is_compiling() and torch.cuda.current_device() != p0.index:
                torch.cuda.set_device(p0)
            return None

        sp.handles.append(model.register_forward_pre_hook(_primary_guard, with_kwargs = True))

    per_dev, comps = {}, {}
    seen = set()
    for i, blk in enumerate(target):
        d = str(devs[assign[i]])
        per_dev[d] = per_dev.get(d, 0) + module_bytes(blk, seen)
    for name, child in model.named_children():
        if name == target_fqn:
            continue
        d = str(module_device(child))
        b = module_bytes(child, seen)
        per_dev[d] = per_dev.get(d, 0) + b
        comps[name] = {"device": d, "bytes": b, "class": type(child).__name__}
    sp.placement = {
        "bytes_per_device": per_dev,
        "gib_per_device": {kk: v / 2**30 for kk, v in per_dev.items()},
        "components": comps,
        "blocks": [str(devs[a]) for a in assign],
        "container": target_fqn,
        "cut": k,
    }
    return sp


def _install(sp, mlist, idx, tr, boundary, leader):
    if boundary == "hook":
        sp.handles.append(
            mlist[idx].register_forward_pre_hook(
                lambda m, a, kw, _t = tr, _l = leader: _t(a, kw, _l), with_kwargs = True
            )
        )
    else:
        original = mlist[idx]
        mlist[idx] = BoundaryShim(original, tr, leader)
        sp.shims.append((mlist, idx, original))


# ============================================================================ device inspection
def device_report(devices):
    devs = [torch.device(d) for d in devices]
    out = {
        "devices": [],
        "can_access_peer": {},
        "p2p_any": False,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
    }
    for d in devs:
        i = d.index or 0
        p = torch.cuda.get_device_properties(i)
        out["devices"].append(
            {
                "index": i,
                "name": p.name,
                "capability": f"{p.major}.{p.minor}",
                "total_gib": p.total_memory / 2**30,
                "sm_count": p.multi_processor_count,
            }
        )
    names = {d["name"] for d in out["devices"]}
    caps = {d["capability"] for d in out["devices"]}
    out["homogeneous_pair"] = len(names) == 1 and len(caps) == 1
    for a in devs:
        for b in devs:
            if a == b:
                continue
            ia, ib = a.index or 0, b.index or 0
            ok = bool(torch.cuda.can_device_access_peer(ia, ib))
            out["can_access_peer"][f"{ia}->{ib}"] = ok
            out["p2p_any"] = out["p2p_any"] or ok
    return out


def copy_bandwidth(
    src,
    dst,
    sizes_mib = (1, 4, 16, 64, 256),
    iters = 20,
):
    s, d = torch.device(src), torch.device(dst)
    rows = {}
    for mb in sizes_mib:
        nbytes = mb << 20
        n = nbytes // 2
        a = torch.empty(n, dtype = torch.bfloat16, device = s)
        b = torch.empty(n, dtype = torch.bfloat16, device = d)
        h = torch.empty(n, dtype = torch.bfloat16, device = "cpu", pin_memory = True)

        def timed(fn):
            for _ in range(10):
                fn()
            sync_all()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            sync_all()
            return (time.perf_counter() - t0) / iters

        direct = timed(lambda: b.copy_(a, non_blocking = True))

        def staged():
            h.copy_(a, non_blocking = False)
            b.copy_(h, non_blocking = False)

        host = timed(staged)
        rows[f"{mb}MiB"] = {
            "bytes": nbytes,
            "direct_ms": direct * 1e3,
            "direct_gib_s": nbytes / 2**30 / direct,
            "host_staged_ms": host * 1e3,
            "host_staged_gib_s": nbytes / 2**30 / host,
        }
        del a, b, h
        free_all()
    return rows


# ================================================================================== the harness
def image_array(out):
    import numpy as np
    img = out.images[0] if hasattr(out, "images") else out[0][0]
    return np.asarray(img).astype("int32")


def render(pipe, args, device):
    g = torch.Generator(device = "cpu").manual_seed(args.seed)
    kw = dict(
        prompt = args.prompt,
        num_inference_steps = args.steps,
        generator = g,
        height = args.size,
        width = args.size,
    )
    if args.guidance is not None:
        kw["guidance_scale"] = args.guidance
    if args.negative_prompt:
        kw["negative_prompt"] = args.negative_prompt
    with torch.no_grad():
        return pipe(**kw)


def dynamo_seconds():
    try:
        from torch._dynamo.utils import cumulative_time_spent_ns
        return sum(cumulative_time_spent_ns.values()) / 1e9
    except Exception:
        return 0.0


def block_compile(model):
    fn = getattr(model, "compile_repeated_blocks", None)
    if not callable(fn):
        raise SystemExit(f"{type(model).__name__} has no compile_repeated_blocks")
    for attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, attr):
            setattr(torch._dynamo.config, attr, 8192)
    fn(fullgraph = True, dynamic = True)


def main():
    ap = argparse.ArgumentParser(
        description = "Two-GPU layer split for a diffusers transformer: correctness and cost."
    )
    ap.add_argument(
        "--repo", default = "Tongyi-MAI/Z-Image-Turbo", help = "diffusers repo id or local path"
    )
    ap.add_argument("--devices", default = "0,1", help = "two CUDA indices, e.g. 0,1")
    ap.add_argument(
        "--prompt",
        default = "a photograph of a city street at night, neon reflections on wet asphalt",
    )
    ap.add_argument("--negative-prompt", default = "")
    ap.add_argument("--steps", type = int, default = 9)
    ap.add_argument("--size", type = int, default = 1024)
    ap.add_argument("--guidance", type = float, default = None)
    ap.add_argument("--seed", type = int, default = 11)
    ap.add_argument("--reps", type = int, default = 5, help = "timed rotations per configuration")
    ap.add_argument("--warm", type = int, default = 1)
    ap.add_argument(
        "--ratio",
        type = float,
        default = None,
        help = "fraction of blocks on the first card; default is memory-balanced",
    )
    ap.add_argument(
        "--boundary",
        default = "shim",
        choices = ("hook", "shim"),
        help = "where the boundary copy runs. 'shim' wraps the boundary block in an "
        "uncompiled module, so the copy stays OUTSIDE the compiled region and "
        "can be timed with CUDA events; 'hook' uses a forward pre-hook, which "
        "torch.compile traces INTO the block and which therefore cannot carry "
        "event instrumentation",
    )
    ap.add_argument(
        "--compile",
        action = "store_true",
        help = "compile_repeated_blocks(fullgraph=True) before timing",
    )
    ap.add_argument(
        "--configs", default = "single,split,split_host,cpu_offload,seq_offload,accel_dispatch"
    )
    ap.add_argument("--dtype", default = "bfloat16", choices = ("bfloat16", "float16", "float32"))
    ap.add_argument("--cache-dir", default = None, help = "TORCHINDUCTOR_CACHE_DIR (persistent)")
    ap.add_argument("--out", default = "two_gpu_split_report.json")
    ap.add_argument("--save-images", default = None, help = "directory for the rendered PNGs")
    args = ap.parse_args()

    if args.cache_dir:
        import os

        p = Path(args.cache_dir)
        p.mkdir(parents = True, exist_ok = True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(p / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(p / "triton")

    if torch.cuda.device_count() < 2:
        print(
            f"NOTE: only {torch.cuda.device_count()} CUDA device(s) visible. The split rows "
            f"need two; the single-GPU and offload rows will still run."
        )
    idx = [int(x) for x in args.devices.split(",")]
    devices = [f"cuda:{i}" for i in idx if i < torch.cuda.device_count()]
    if not devices:
        raise SystemExit("no usable CUDA devices")
    primary = devices[0]
    dtype = getattr(torch, args.dtype)

    # Determinism BEFORE the first compile: without it inductor redraws its reduction block size
    # per compilation and a split render can differ from a single-GPU one for reasons that have
    # nothing to do with the split.
    torch.use_deterministic_algorithms(True, warn_only = True)
    torch.utils.deterministic.fill_uninitialized_memory = False

    report = {
        "tool": "bench_two_gpu_split.py",
        "version": __version__,
        "argv": sys.argv,
        "args": vars(args),
        "failures": [],
    }
    report["environment"] = device_report(devices)
    print(_banner("ENVIRONMENT"))
    for d in report["environment"]["devices"]:
        print(
            f"  cuda:{d['index']}  {d['name']}  sm_{d['capability'].replace('.', '')}  "
            f"{d['total_gib']:.1f} GiB  {d['sm_count']} SMs"
        )
    print(
        f"  peer access: {report['environment']['can_access_peer']}  "
        f"(P2P {'available' if report['environment']['p2p_any'] else 'NOT available: the '
                'driver will stage copies through host memory'})"
    )
    print(
        f"  homogeneous pair: {report['environment']['homogeneous_pair']}"
        + (
            ""
            if report["environment"]["homogeneous_pair"]
            else "  <- mixed cards: bit-identity is NOT expected"
        )
    )
    print(
        f"  torch {torch.__version__}  cuda {getattr(torch.version, 'cuda', None)}  "
        f"{platform.system()}"
    )

    if len(devices) >= 2:
        print(_banner("COPY BANDWIDTH cuda:%d -> cuda:%d" % (idx[0], idx[1])))
        report["copy_bandwidth"] = copy_bandwidth(devices[0], devices[1])
        print(f"  {'size':>8}  {'direct':>12}  {'GiB/s':>9}  {'host-staged':>12}  {'GiB/s':>9}")
        for k, v in report["copy_bandwidth"].items():
            print(
                f"  {k:>8}  {v['direct_ms']:10.3f}ms  {v['direct_gib_s']:9.1f}  "
                f"{v['host_staged_ms']:10.3f}ms  {v['host_staged_gib_s']:9.1f}"
            )

    # ---- load ---------------------------------------------------------------------------------
    from diffusers import DiffusionPipeline

    print(_banner("LOADING"))
    t0 = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained(args.repo, torch_dtype = dtype)
    pipe.to(primary)
    pipe.set_progress_bar_config(disable = True)
    report["load_seconds"] = time.perf_counter() - t0
    print(f"  {args.repo} in {report['load_seconds']:.1f}s")

    model = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
    if model is None:
        raise SystemExit("pipeline has neither .transformer nor .unet")
    c = census(model)
    report["census"] = {k: v for k, v in c.items() if k != "containers_raw"}
    print(_banner("MODEL"))
    print(f"  {c['class']}  repeated blocks {c['block_classes']}")
    for cc in c["containers"]:
        print(f"    {cc['fqn']:<20s} {cc['n']:>3d} blocks  {cc['bytes'] / 2**30:8.3f} GiB")
    print(f"    {'non-block':<20s}     {' ':>3s}       " f"{c['non_block_bytes'] / 2**30:8.3f} GiB")
    print(f"    {'TOTAL':<20s}     {' ':>3s}       {c['total_bytes'] / 2**30:8.3f} GiB")
    for name, comp in pipe.components.items():
        if isinstance(comp, nn.Module) and comp is not model:
            print(
                f"    component {name:<16s} {module_bytes(comp) / 2**30:8.3f} GiB "
                f"on {module_device(comp)}"
            )

    if args.compile:
        print(_banner("COMPILE"))
        d0 = dynamo_seconds()
        t0 = time.perf_counter()
        block_compile(model)
        render(pipe, args, primary)
        sync_all()
        report["compile"] = {
            "wall_seconds": time.perf_counter() - t0,
            "dynamo_seconds": dynamo_seconds() - d0,
        }
        print(
            f"  compile_repeated_blocks(fullgraph=True) + first render: "
            f"{report['compile']['wall_seconds']:.1f}s wall, "
            f"{report['compile']['dynamo_seconds']:.1f}s in dynamo/inductor"
        )

    # ---- configurations -----------------------------------------------------------------------
    want = [x for x in args.configs.split(",") if x]
    two = len(devices) >= 2
    cfgs = []
    for name in want:
        if name in ("split", "split_host", "accel_dispatch") and not two:
            report["failures"].append({"config": name, "error": "needs two visible GPUs"})
            continue
        cfgs.append(name)

    results = {}
    reference = None
    print(_banner("CORRECTNESS + TIMING"))
    for name in cfgs:
        entry = {"config": name, "seconds": [], "setup_seconds": None}
        sp = None
        try:
            d0 = dynamo_seconds()
            t0 = time.perf_counter()
            sp = _setup(name, pipe, model, devices, args)
            entry["setup_seconds"] = time.perf_counter() - t0
            if sp is not None:
                entry["placement"] = sp.placement
                entry["reasoning"] = sp.reasoning
            for _ in range(args.warm):
                render(pipe, args, primary)
            sync_all()
            entry["compile_seconds"] = dynamo_seconds() - d0
            out = render(pipe, args, primary)
            sync_all()
            arr = image_array(out)
            if reference is None:
                reference = arr
                entry["is_reference"] = True
            diff = abs(arr - reference)
            entry["max_abs_diff_vs_single"] = int(diff.max())
            entry["n_pixels_differing"] = int((arr != reference).sum())
            entry["bit_identical"] = entry["max_abs_diff_vs_single"] == 0
            if args.save_images:
                Path(args.save_images).mkdir(parents = True, exist_ok = True)
                out.images[0].save(Path(args.save_images) / f"{name}.png")
            for _ in range(args.reps):
                sync_all()
                t0 = time.perf_counter()
                render(pipe, args, primary)
                sync_all()
                entry["seconds"].append(time.perf_counter() - t0)
            entry["memory"] = {
                f"cuda:{i}": {
                    "allocated_gib": torch.cuda.memory_allocated(i) / 2**30,
                    "reserved_gib": torch.cuda.memory_reserved(i) / 2**30,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(i) / 2**30,
                }
                for i in idx
                if i < torch.cuda.device_count()
            }
            if sp is not None:
                entry["boundary"] = sp.transfer_stats()
            entry["p50"] = statistics.median(entry["seconds"])
            entry["min"] = min(entry["seconds"])
            entry["stdev"] = (
                statistics.stdev(entry["seconds"]) if len(entry["seconds"]) > 1 else 0.0
            )
            entry["ok"] = True
        except SkipConfig as exc:
            entry["ok"] = False
            entry["skipped"] = True
            entry["error"] = str(exc)
            print(f"  {name:<18s} SKIPPED: {exc}")
        except Exception as exc:
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()[-3000:]
            print(f"  {name:<18s} FAILED: {entry['error']}")
        finally:
            try:
                _teardown(name, pipe, model, devices, sp)
            except Exception as exc:
                report["failures"].append({"config": name, "teardown": str(exc)})
            free_all()
        results[name] = entry
        if entry.get("ok"):
            print(
                f"  {name:<18s} p50 {entry['p50']:7.3f}s  min {entry['min']:7.3f}s  "
                f"bit-identical={entry['bit_identical']}  "
                f"setup {entry['setup_seconds']:6.2f}s  "
                f"compile {entry.get('compile_seconds', 0):6.1f}s"
            )

    report["results"] = results
    _print_table(report, args)
    Path(args.out).write_text(json.dumps(_jsonable(report), indent = 2))
    print(f"\nwrote {Path(args.out).resolve()}")
    return 0


def _setup(name, pipe, model, devices, args):
    primary = devices[0]
    if name == "single":
        pipe.to(primary)
        return None
    if name in ("split", "split_host"):
        return split_transformer(
            model,
            devices,
            ratio = args.ratio,
            stage = "host" if name == "split_host" else "direct",
            boundary = args.boundary,
            instrument = (args.boundary == "shim" or not args.compile),
        )
    if name == "accel_dispatch":
        # accelerate attaches an AlignDevicesHook to every submodule in the map, and that hook is
        # decorated torch.compiler.disable, so Dynamo cannot trace through it: with regional
        # compile on, dispatch_model and compile_repeated_blocks(fullgraph=True) are mutually
        # exclusive. That is a result, not an error, so say it plainly.
        if getattr(args, "compile", False):
            raise SkipConfig(
                "accelerate dispatch_model cannot be combined with "
                "compile_repeated_blocks(fullgraph=True): its AlignDevicesHook is "
                "torch.compiler.disable'd, so Dynamo refuses with 'Skip calling "
                "torch.compiler.disable()d function'. Re-run without --compile to measure "
                "this row."
            )
        from accelerate import dispatch_model

        c = census(model)
        containers = c["containers_raw"]
        fqn, target = max(containers, key = lambda t: sum(c["block_bytes"][t[0]]))
        fixed = c["non_block_bytes"] + sum(
            sum(c["block_bytes"][f]) for f, _ in containers if f != fqn
        )
        k, _ = plan_cut(c["block_bytes"][fqn], fixed, args.ratio)
        i0, i1 = torch.device(devices[0]).index, torch.device(devices[1]).index
        dm = {n: i0 for n, _ in model.named_children() if n != fqn}
        for i in range(len(target)):
            dm[f"{fqn}.{i}"] = i0 if i < k else i1
        # accelerate requires the map to be EXHAUSTIVE. A map built from named_children() misses
        # the model's own top-level parameters and dispatch_model refuses outright with "The
        # device_map provided does not give any device for the following parameters: ...". Ours
        # does not need this step because it moves whatever it does not cut.
        for pname, _ in list(model.named_parameters(recurse = False)) + list(
            model.named_buffers(recurse = False)
        ):
            dm[pname] = i0
        dispatch_model(model, device_map = dm, main_device = torch.device(devices[0]))
        sp = Split()
        sp.placement = {
            "mechanism": "accelerate.dispatch_model",
            "cut": k,
            "device_map_entries": len(dm),
        }
        return sp
    if name == "cpu_offload":
        pipe.enable_model_cpu_offload(device = primary)
        return None
    if name == "seq_offload":
        # Sequential offload and regional compile are mutually exclusive: Dynamo reaches
        # accelerate's AlignDevicesHook.pre_forward, which is decorated torch.compiler.disable,
        # and refuses with "Unsupported: Skip calling torch.compiler.disable()d function". The
        # caller is told rather than shown a traceback, because the incompatibility is the result.
        if getattr(args, "compile", False):
            raise SkipConfig(
                "sequential CPU offload cannot be combined with "
                "compile_repeated_blocks(fullgraph=True): accelerate's AlignDevicesHook is "
                "torch.compiler.disable'd, so Dynamo cannot trace through it. Re-run without "
                "--compile to measure this row."
            )
        pipe.enable_sequential_cpu_offload(device = primary)
        return None
    raise SystemExit(f"unknown config {name!r}")


class SkipConfig(Exception):
    """This configuration cannot be measured here, and the reason is itself the finding."""


def _teardown(name, pipe, model, devices, sp):
    primary = devices[0]
    if name in ("split", "split_host") and sp is not None:
        sp.undo(model, primary)
    elif name == "accel_dispatch":
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(model, recurse = True)
        model.to(primary)
        if hasattr(model, "hf_device_map"):
            try:
                delattr(model, "hf_device_map")
            except Exception:
                pass
    elif name in ("cpu_offload", "seq_offload"):
        try:
            pipe.remove_all_hooks()
        except Exception:
            pass
        pipe.to(primary)


def _banner(text):
    return f"\n{'=' * 78}\n{text}\n{'=' * 78}"


def _print_table(report, args):
    res = report["results"]
    base = res.get("single", {})
    b50 = base.get("p50")
    print(_banner("SUMMARY"))
    hdr = (
        f"  {'configuration':<18s} {'p50 s':>8s} {'min s':>8s} {'vs single':>10s} "
        f"{'bit-id':>7s} {'compile s':>10s} {'card0 GiB':>10s} {'card1 GiB':>10s} "
        f"{'boundary MiB/fwd':>17s} {'boundary ms/fwd':>16s}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, e in res.items():
        if not e.get("ok"):
            print(f"  {name:<18s} {'FAILED':>8s}  {str(e.get('error'))[:60]}")
            continue
        mem = e.get("memory", {})
        keys = sorted(mem)
        c0 = mem.get(keys[0], {}).get("peak_allocated_gib", float("nan")) if keys else float("nan")
        c1 = (
            mem.get(keys[1], {}).get("peak_allocated_gib", float("nan"))
            if len(keys) > 1
            else float("nan")
        )
        bmib = sum(t["bytes_per_forward"] for t in e.get("boundary", [])) / 2**20
        bms = sum(t["ms_per_forward"] for t in e.get("boundary", []))
        ratio = (e["p50"] / b50) if b50 else float("nan")
        print(
            f"  {name:<18s} {e['p50']:8.3f} {e['min']:8.3f} {ratio:9.3f}x "
            f"{str(e['bit_identical']):>7s} {e.get('compile_seconds', 0):10.1f} "
            f"{c0:10.3f} {c1:10.3f} {bmib:17.2f} {bms:16.3f}"
        )
    steps = args.steps
    for name, e in res.items():
        if not e.get("ok") or not e.get("boundary"):
            continue
        bms = sum(t["ms_per_forward"] for t in e["boundary"])
        if bms and e["p50"]:
            print(
                f"\n  {name}: boundary transfer is {bms * steps / 1000:.4f}s of a "
                f"{e['p50']:.3f}s render ({bms * steps / 10 / e['p50']:.2f}% of wall clock) "
                f"over {steps} steps"
            )
            for t in e["boundary"]:
                print(
                    f"    {t['name']}: {t['bytes_per_forward'] / 2**20:8.2f} MiB/forward, "
                    f"{t['ms_per_forward']:7.3f} ms/forward, tensors "
                    f"{ {k: v[0] for k, v in t['tensors'].items()} }"
                )
    if not report["environment"]["homogeneous_pair"]:
        print(
            "\n  NOTE: the two cards are different models. A bit-identical result is not "
            "expected:\n        different SM counts change the tile and split-k choice inside "
            "cuBLAS,\n        which changes the reduction order and therefore the last bits."
        )
    if not report["environment"]["p2p_any"]:
        print(
            "\n  NOTE: no peer-to-peer access between these cards. The 'split' row already "
            "reflects\n        the driver's host-staged path, and 'split_host' is the explicit "
            "version of it."
        )


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if k != "containers_raw"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, torch.dtype) or isinstance(obj, torch.device):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130)
