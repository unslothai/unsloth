# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Shared pieces of the diffusion samplers' fast path: in-loop cancellation and throttled canvas frames, the
active-block-only output head, and regional compile of the backbone for inference."""

import contextlib
import time

import torch

from .diffusion_loss import (
    _decoder_layer_lists,
    _warm_rotary_caches,
    fusable_head,
    head_returns_hidden,
)

__all__ = [
    "DiffusionGenerationCancelled",
    "Progress",
    "make_progress",
    "BlockHead",
    "GraphedForward",
    "inference_compile",
    "resolve_compile",
]


class DiffusionGenerationCancelled(RuntimeError):
    """Raised out of generate_text when stop_callback() returned True mid-generation."""


class Progress:
    """Cancellation checks and canvas frames for one generate_text call.

    ``check()`` raises DiffusionGenerationCancelled once stop_callback() returns True. ``frame(...)`` hands
    on_frame a host copy of the generated region at most once per ``interval`` seconds (always when
    ``force``): the device copy is queued non-blocking into a pinned buffer and delivered on the next call,
    so the sampler never waits on it."""

    def __init__(self, on_frame, stop_callback, interval, mask_id, budget):
        self.on_frame = on_frame
        self.stop_callback = stop_callback
        self.interval = max(float(interval or 0.0), 0.0)
        self.mask_id = mask_id
        self.budget = int(budget)
        self._last = -float("inf")
        self._pending = None
        self.closed = False

    def check(self):
        if self.stop_callback is not None and self.stop_callback():
            raise DiffusionGenerationCancelled("Unsloth: diffusion generation cancelled.")

    def _emit(self, meta, tokens):
        if not self.closed:
            self.on_frame({**meta, "tokens": tokens, "mask_id": self.mask_id})

    def _flush(self, wait):
        if self._pending is None:
            return
        meta, buffer, event = self._pending
        if event is not None and not wait and not event.query():
            return
        if event is not None:
            event.synchronize()
        self._pending = None
        self._emit(meta, buffer.tolist())

    def frame(
        self,
        block,
        step,
        total,
        generated,
        force = False,
    ):
        """``generated`` is the first row's generated region ([L] or [1, L]); padding past the budget is cut."""
        if self.on_frame is None or self.closed:
            return
        self._flush(wait = force)
        now = time.monotonic()
        if not force and now - self._last < self.interval:
            return
        self._last = now
        row = generated.reshape(-1, generated.shape[-1])[0, : self.budget].detach()
        meta = {"block": int(block), "step": int(step), "total": int(total)}
        if row.device.type != "cuda":
            self._emit(meta, row.tolist())
            return
        # A one-row id snapshot: pinned host buffer + event, so the GPU never waits for the host.
        buffer = torch.empty(row.shape, dtype = row.dtype, pin_memory = True)
        buffer.copy_(row, non_blocking = True)
        event = torch.cuda.Event()
        event.record()
        self._pending = (meta, buffer, event)
        if force:
            self._flush(wait = True)

    def close(self):
        """Deliver the last queued frame, then never call on_frame again."""
        if self.on_frame is not None and not self.closed:
            self._flush(wait = True)
        self.closed = True


def make_progress(
    on_frame,
    stop_callback,
    interval = 0.1,
    mask_id = None,
    budget = 0,
):
    return Progress(on_frame, stop_callback, interval, mask_id, budget)


class GraphedForward:
    """The backbone's eager forward captured into a CUDA Graph per input shape and replayed every step.

    Same kernels as eager, so the same numbers, without per-kernel launch overhead or a dynamo trace: a new
    canvas length costs two warm-up forwards and one capture. A forward that syncs with the host (an expert
    loop over ``torch.where``) cannot be captured: the first failure turns graphs off for this call."""

    def __init__(
        self,
        function,
        max_graphs = 2,
        owner = None,
    ):
        self.function = function
        self.max_graphs = max_graphs
        self.graphs = {}
        self.pool = None
        # Remembered on the model, so later calls skip straight to eager instead of re-trying the capture.
        self.owner = owner
        self.failed = None if owner is None else owner.__dict__.get("_unsloth_graph_capture_failed")

    def _capture(self, x):
        static_x = x.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(
                2
            ):  # warm-up outside capture: lazy buffers, cuBLAS handles, autotuned kernels
                self.function(static_x)
        torch.cuda.current_stream().wait_stream(stream)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool = self.pool):
            static_out = self.function(static_x)
        return graph, static_x, static_out

    def __call__(self, x):
        if self.failed is not None:
            return self.function(x)
        key = (tuple(x.shape), x.dtype, x.device)
        entry = self.graphs.pop(key, None)
        if entry is None:
            try:
                entry = self._capture(x)
            except DiffusionGenerationCancelled:
                # stop_callback's forward hook fired during warm-up / capture: not a capture failure.
                self.graphs.clear()
                raise
            except Exception as error:  # capture-unsafe forward: eager from here on
                self.failed = f"{type(error).__name__}: {error}"[:300]
                if self.owner is not None:
                    self.owner.__dict__["_unsloth_graph_capture_failed"] = self.failed
                self.graphs.clear()
                torch.cuda.synchronize()
                return self.function(x)
            while len(self.graphs) >= self.max_graphs:
                self.graphs.pop(next(iter(self.graphs)))
        self.graphs[key] = entry  # most recently used last
        graph, static_x, static_out = entry
        static_x.copy_(x)
        graph.replay()
        return static_out


class BlockHead:
    """Runs the backbone over the full canvas but projects only the requested rows to the vocabulary.

    With a plain nn.Linear head the model returns final hidden states and ``logits(x, start, end)`` applies the
    head to ``hidden[:, start:end]`` only: no [B, L, V] tensor. Any other head (tied, scaled, LoRA, quantized)
    keeps the model's own logits and slices them, so the result is the same either way. ``graphs`` replays the
    backbone from CUDA Graphs (``GraphedForward``)."""

    def __init__(
        self,
        base,
        cuda_graphs = False,
        graphs = False,
    ):
        self.base = base
        self.head = head = fusable_head(base, base)
        self.cuda_graphs = cuda_graphs

        def forward(x):
            if head is None:
                return base(x).logits
            with head_returns_hidden(head):
                return base(x).logits

        # Not a bound method: a BlockHead -> GraphedForward -> BlockHead cycle would keep the graphs and their
        # memory pool alive after the call until a full gc pass.
        self._forward = forward
        self.graphed = GraphedForward(forward, owner = base) if graphs else None

    def logits(self, x, start, end):
        if self.cuda_graphs:
            # Each step reuses the compiled graphs' output buffers; the previous step's outputs are dead by now.
            torch.compiler.cudagraph_mark_step_begin()
        out = self.graphed(x) if self.graphed is not None else self._forward(x)
        out = out[:, start:end]
        return out if self.head is None else self.head(out)


def resolve_compile(compile, device):
    """How the backbone runs on this device. "auto" / True: eager kernels replayed from CUDA Graphs ("graphs",
    same numbers as eager); False / None, or anything off CUDA: plain eager. Other strings pick torch.compile:
    "cudagraphs" (dynamo backend, eager numerics, re-traced per canvas length) or an inductor mode such as
    "default" / "reduce-overhead" (fused kernels, so greedy output can drift from eager)."""
    if compile is None or compile is False or device.type != "cuda":
        return None
    if compile is True or compile == "auto":
        return "graphs"
    return str(compile)


@contextlib.contextmanager
def inference_compile(base, mode):
    """Compile each decoder layer for this generate call and put the previous call path back afterwards, so a
    later training step never runs an inference-mode (CUDA Graph) layer. The compiled wrapper is cached on the
    layer, so dynamo reuses its graphs across calls. mode None does nothing."""
    if mode is None or mode == "graphs":
        yield 0
        return
    _warm_rotary_caches(base)
    saved = []
    for _, layers in _decoder_layer_lists(base):
        for layer in layers:
            cache = layer.__dict__.setdefault("_unsloth_infer_compiled", {})
            if mode not in cache:
                if mode == "cudagraphs":
                    # Eager kernels replayed from CUDA Graphs: no inductor fusion, so eager numerics.
                    options = {"backend": "cudagraphs", "dynamic": False}
                else:
                    options = {"mode": None if mode == "default" else mode, "dynamic": True}
                cache[mode] = torch.compile(layer._call_impl, **options)
            saved.append((layer, layer._compiled_call_impl))
            layer._compiled_call_impl = cache[mode]
    try:
        yield len(saved)
    finally:
        for layer, previous in saved:
            layer._compiled_call_impl = previous
