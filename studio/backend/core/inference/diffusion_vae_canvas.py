# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Composited-canvas tiled decode for VAEs that tile like diffusers' MiniMax-H3 VAEs (``_split_tiles`` lays the
tiles, ``_blend`` ramps the overlaps, ``_stitch_tiles`` joins them, each tile is ``decoder(post_quant_conv(tile))``):
the H3-Image ``_decode`` and the H3 video ``_decode_clip``.

The stock stitch blends every tile with its RAW upper and left neighbours, so where four tiles meet the earlier blends
are overwritten (a constant 0 / 1 / 2 / 3 grid reads 2.0 at an interior corner pixel that belongs to tile 0). Here
every tile is added into one preallocated output with the separable stock ramps (same formula, same dtype as
``_blend``): a normalised overlap-add. Outside the four-tile corners that is bit-identical to the stock stitch; in
them it is the overlap-add the ramps describe. It also drops the per-row and final concatenations.

Levers, each with its own switch (``0`` turns it off):
- canvas: ``UNSLOTH_DIFFUSION_VAE_CANVAS`` (off = the stock stitch);
- tile batch: ``UNSLOTH_DIFFUSION_VAE_TILE_BATCH`` (several equal tiles per decoder call, sized from free VRAM;
  ``N`` forces N);
- tile graphs: ``UNSLOTH_DIFFUSION_VAE_TILE_GRAPHS`` (one CUDA graph per tile batch shape, replicas on their own
  streams and pools, one replica on ROCm; raced in place on the real tiles against eager, eager kept unless a graph
  arm is faster by the margin, each replica checked ``torch.equal`` to eager first; ``1`` / ``N`` force N replicas);
- half decoder: ``UNSLOTH_DIFFUSION_VAE_HALF`` (fp16 block GEMMs and attention for ViT decoders whose norms,
  scales and residual stream are pinned fp32; every decode is probed for non-finite or flat output, which re-decodes
  in fp32 and keeps fp32 for the load).
"""

from __future__ import annotations

import hashlib
import inspect
import os
import threading
import types
from typing import Any, Optional

CANVAS_ENV = "UNSLOTH_DIFFUSION_VAE_CANVAS"
TILE_BATCH_ENV = "UNSLOTH_DIFFUSION_VAE_TILE_BATCH"
TILE_GRAPHS_ENV = "UNSLOTH_DIFFUSION_VAE_TILE_GRAPHS"
HALF_ENV = "UNSLOTH_DIFFUSION_VAE_HALF"

LEVER_CANVAS = "vae_canvas"
LEVER_TILE_BATCH = "vae_tile_batch"
LEVER_TILE_GRAPHS = "vae_tile_graphs"
LEVER_HALF = "vae_fp16"

# sha256[:16] of the stock ``_blend`` / ``_stitch_tiles`` sources the canvas reproduces (diffusers 0.41-0.42 and the
# vendored H3-Image copy, which carry the same text).
_BLEND_SHA = frozenset({"2b6df9fb4c3a9c96"})
_STITCH_SHA = frozenset({"ae481626afa30528"})

TILE_BATCH_MAX = 16
TILE_BATCH_FREE_FRACTION = 0.5
TILE_BYTES_FALLBACK = 256 * 2**20

# Replica counts raced on CUDA (concurrent replay can hang on ROCm: one replica there).
GRAPH_REPLICAS = (1, 2, 4)
GRAPH_REPLICAS_ROCM = (1,)
# A graph arm must beat eager by this fraction; between graph arms within it, fewer replicas win.
GRAPH_MARGIN = 0.02
# Timed decoder calls per race arm (at least one per replica); eager also gets one untimed warm-up call first.
GRAPH_RACE_CALLS = 3
# Kept graph pools (one per replica, reserved between renders): at most this many bytes or this share of the card.
GRAPH_POOL_MAX_BYTES = 1 << 30
GRAPH_POOL_MAX_FRACTION = 0.04

_OFF = ("0", "off", "false", "no")
_ON = ("1", "on", "true", "yes")
_SHA_CACHE: dict = {}
_LOCK = threading.Lock()


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip().lower()


def canvas_enabled() -> bool:
    return _env(CANVAS_ENV) not in _OFF


def tile_batch_setting() -> Optional[int]:
    """None = auto; 1 = off; N = forced."""
    raw = _env(TILE_BATCH_ENV)
    if raw in _OFF:
        return 1
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return None


def tile_graphs_setting() -> Optional[int]:
    """None = race; 0 = off; N = forced N replicas."""
    raw = _env(TILE_GRAPHS_ENV)
    if raw in _OFF:
        return 0
    if raw in _ON:
        return 1
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return None


def half_enabled() -> bool:
    return _env(HALF_ENV) not in _OFF


def _source_sha(fn: Any) -> Optional[str]:
    key = getattr(fn, "__code__", fn)
    hit = _SHA_CACHE.get(key)
    if hit is None:
        try:
            hit = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]
        except Exception:  # noqa: BLE001
            hit = ""
        _SHA_CACHE[key] = hit
    return hit or None


def stock_tiling(vae: Any) -> bool:
    """Whether ``vae`` tiles with the stock MiniMax loop the canvas reproduces (own overrides on the instance and
    edited sources keep stock)."""
    cls = type(vae)
    if not all(
        callable(getattr(cls, n, None)) for n in ("_split_tiles", "_blend", "_stitch_tiles")
    ):
        return False
    if any(n in getattr(vae, "__dict__", {}) for n in ("_blend", "_stitch_tiles", "_split_tiles")):
        return False
    if not hasattr(vae, "decoder") or not hasattr(vae, "post_quant_conv"):
        return False
    return _source_sha(cls._blend) in _BLEND_SHA and _source_sha(cls._stitch_tiles) in _STITCH_SHA


def axis_ok(starts: list, lengths: list, overlaps: list, total: int, ratio: int) -> bool:
    """The layout the canvas handles: latent-aligned tiles from 0 to ``total``, each next tile starting where the
    previous one's trailing overlap begins, overlaps > 0 and never three tiles over one pixel."""
    n = len(starts)
    if n == 0 or len(lengths) != n or len(overlaps) != n - 1:
        return False
    if starts[0] != 0 or starts[-1] + lengths[-1] != total:
        return False
    if any(s % ratio or length % ratio or length <= 0 for s, length in zip(starts, lengths)):
        return False
    for k in range(n - 1):
        e = overlaps[k]
        if e <= 0:
            return False
        if e > min(lengths[k], lengths[k + 1]) or starts[k + 1] != starts[k] + lengths[k] - e:
            return False
        if k > 0 and overlaps[k - 1] + e > lengths[k]:
            return False
    return True


def axis_ramps(lengths: list, overlaps: list, dtype: Any, device: Any) -> list:
    """Per-tile weights along one axis with the stock ``_blend`` ramps: the incoming tile ``p / E``, the outgoing
    ``1 - p / E`` (computed exactly as ``_blend`` does, in the tile dtype), 1 elsewhere."""
    import torch

    out = []
    n = len(lengths)
    for k, length in enumerate(lengths):
        w = torch.ones(length, dtype = dtype, device = device)
        if k > 0:
            e = overlaps[k - 1]
            positions = torch.arange(e, device = device, dtype = dtype)
            w[:e] = positions / e
        if k < n - 1:
            e = overlaps[k]
            positions = torch.arange(e, device = device, dtype = dtype)
            w[length - e :] = 1 - positions / e
        out.append(w)
    return out


class Canvas:
    """One preallocated output; ``add`` accumulates a tile with its separable ramp weights."""

    __slots__ = ("out", "ys", "xs", "wy", "wx", "_w", "ylen", "xlen")

    def __init__(self, like: Any, height: int, width: int, y: tuple, x: tuple) -> None:
        import torch

        ys, ylen, yov = y
        xs, xlen, xov = x
        self.out = torch.zeros(
            (*like.shape[:-2], height, width), dtype = like.dtype, device = like.device
        )
        self.ys, self.xs, self.ylen, self.xlen = ys, xs, ylen, xlen
        self.wy = axis_ramps(ylen, yov, like.dtype, like.device)
        self.wx = axis_ramps(xlen, xov, like.dtype, like.device)
        self._w: dict = {}

    def weight(self, i: int, j: int) -> Any:
        w = self._w.get((i, j))
        if w is None:
            w = self._w[(i, j)] = self.wy[i].view(-1, 1) * self.wx[j].view(1, -1)
        return w

    def add(self, i: int, j: int, tile: Any) -> None:
        y, x = self.ys[i], self.xs[j]
        view = self.out[..., y : y + self.ylen[i], x : x + self.xlen[j]]
        view.add_(tile * self.weight(i, j))


def oracle_stitch(tiles: list, y: tuple, x: tuple) -> Any:
    """Reference overlap-add of a full tile grid (tests, benches): ``Canvas`` over ``tiles[i][j]``."""
    first = tiles[0][0]
    ys, ylen, _ = y
    xs, xlen, _ = x
    canvas = Canvas(first, ys[-1] + ylen[-1], xs[-1] + xlen[-1], y, x)
    for i, row in enumerate(tiles):
        for j, t in enumerate(row):
            canvas.add(i, j, t)
    return canvas.out


def layout(vae: Any, z: Any) -> Optional[tuple]:
    """((ys, ylens, yoverlaps), (xs, xlens, xoverlaps), ratio) in pixels for ``z``, or None when the canvas does
    not cover this layout."""
    ratio = int(vae.spatial_compression_ratio)
    height = z.shape[-2] * ratio
    width = z.shape[-1] * ratio
    y = vae._split_tiles(height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height)
    x = vae._split_tiles(width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width)
    if not axis_ok(*y, height, ratio) or not axis_ok(*x, width, ratio):
        return None
    return tuple(map(list, y)), tuple(map(list, x)), ratio


def _decoder_tile_bytes(vae: Any, tile: Any) -> int:
    """Working set of one tile through a ViT decoder: the widest per-token activation (fp32) a few times over."""
    decoder = getattr(vae, "decoder", None)
    blocks = getattr(decoder, "transformer_blocks", None)
    try:
        first = blocks[0]
        widest = 0
        for m in first.modules():
            out = getattr(m, "out_features", None)
            if isinstance(out, int):
                widest = max(widest, out)
        tokens = 1
        for s in tile.shape[2:]:
            tokens *= int(s)
        tokens += int(getattr(decoder, "num_register_tokens", 0)) + 1
        ratio = int(vae.spatial_compression_ratio)
        pixels = int(getattr(decoder, "out_channels", 3)) * tokens * ratio * ratio
        if widest <= 0:
            return TILE_BYTES_FALLBACK
        return int(tokens * widest * 4 * 4 + pixels * 4 * 3)
    except Exception:  # noqa: BLE001
        return TILE_BYTES_FALLBACK


def tile_batch_size(vae: Any, tile: Any, count: int) -> int:
    import torch

    forced = tile_batch_setting()
    if forced is not None:
        return max(1, min(forced, count))
    if tile.device.type != "cuda" or not vae.__dict__.get("_unsloth_canvas_batch_ok", True):
        return 1
    try:
        from .diffusion_memory import user_budget_reserve_bytes
        free, total = torch.cuda.mem_get_info(tile.device)
        free = max(0, int(free) - user_budget_reserve_bytes(tile.device, total))
    except Exception:  # noqa: BLE001
        return 1
    per = _decoder_tile_bytes(vae, tile) * max(1, int(tile.shape[0]))
    fit = int(free * TILE_BATCH_FREE_FRACTION // max(1, per))
    # The last size that fit stays while it still fits: a steady size keeps one race / recording per tile shape.
    sizes = vae.__dict__.setdefault("_unsloth_canvas_batches", {})
    key = (tuple(tile.shape), count)
    kept = sizes.get(key)
    if kept is not None and kept <= fit:
        return kept
    batch = max(1, min(TILE_BATCH_MAX, fit, count))
    calls = -(-count // batch)
    batch = sizes[key] = -(-count // calls)
    return batch


def _is_oom(exc: BaseException) -> bool:
    import torch

    oom = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", None)
    if oom is not None and isinstance(exc, oom):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _is_rocm() -> bool:
    try:
        from .rocm_bf16 import is_rocm_torch
        return bool(is_rocm_torch())
    except Exception:  # noqa: BLE001
        import torch
        return bool(getattr(torch.version, "hip", None))


class _Replica:
    __slots__ = ("graph", "stream", "static_in", "static_out", "done", "consumed", "bytes")


class TileGraphs:
    """CUDA graphs of ``fn`` at one input shape: replica ``r`` owns a stream, a private pool, its static input and
    output. ``launch`` copies the group in and replays on the replica's stream; the caller's stream waits for it,
    reads the output, then ``release`` lets the replica be reused."""

    def __init__(self, fn: Any, shape: tuple, dtype: Any, device: Any) -> None:
        self.fn, self.shape, self.dtype, self.device = fn, tuple(shape), dtype, device
        self.replicas: list = []
        self.failed: Optional[str] = None
        self.replays = 0

    @property
    def pool_bytes(self) -> int:
        return sum(r.bytes for r in self.replicas)

    def grow(
        self,
        n: int,
        sample: Any,
        reference: Any = None,
    ) -> Optional[bool]:
        """Capture replicas up to ``n``; each new one is replayed on ``sample`` and must equal ``reference``
        (the eager output of the same input) bit for bit. None: refused for now, nothing recorded."""
        import torch

        from .diffusion_cuda_graph import (
            CAPTURE_MODE,
            CaptureRefused,
            _capturing,
            refuse_if_exhausted,
            retire_failed_capture,
        )

        while len(self.replicas) < n and self.failed is None:
            rep = _Replica()
            graph = None
            try:
                refuse_if_exhausted()
                main = torch.cuda.current_stream(self.device)
                stream = torch.cuda.Stream(device = self.device)
                stream.wait_stream(main)
                before = torch.cuda.memory_reserved(self.device)
                with torch.cuda.stream(stream):
                    static_in = torch.empty(self.shape, dtype = self.dtype, device = self.device)
                    static_in.copy_(sample)
                    # Warm-up on the replica's own stream: its cuBLAS handle and workspace are its own.
                    self.fn(static_in)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                # A private pool per replica: replicas replay concurrently, so they must not share memory.
                pool = torch.cuda.graph_pool_handle()
                with torch.cuda.stream(stream), _capturing():
                    try:
                        graph.capture_begin(pool = pool, capture_error_mode = CAPTURE_MODE)
                        try:
                            static_out = self.fn(static_in)
                        finally:
                            graph.capture_end()
                    except BaseException as exc:
                        retire_failed_capture(graph, pool, exc)
                        graph = None
                        raise
                main.wait_stream(stream)
                rep.bytes = max(0, torch.cuda.memory_reserved(self.device) - before)
                rep.graph, rep.stream, rep.static_in, rep.static_out = (
                    graph,
                    stream,
                    static_in,
                    static_out,
                )
                rep.done, rep.consumed = None, None
                if reference is not None:
                    with torch.cuda.stream(stream):
                        graph.replay()
                    main.wait_stream(stream)
                    if not torch.equal(static_out, reference):
                        raise RuntimeError("graph replay differs from the eager decode")
                self.replicas.append(rep)
            except CaptureRefused:
                # Another thread is recording, or replica work did not drain: nothing recorded, retry later.
                return None
            except Exception as exc:  # noqa: BLE001 - the decode stays eager
                self.failed = f"{type(exc).__name__}: {exc}"[:200]
                self.drop()
                return False
        return self.failed is None

    def keep(self, n: int) -> None:
        del self.replicas[n:]

    def drop(self) -> None:
        if self.replicas:
            import torch
            torch.cuda.synchronize(self.device)
        self.replicas = []

    def launch(self, k: int, inputs: list, ready: Any) -> tuple:
        import torch

        rep = self.replicas[k % len(self.replicas)]
        main = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(rep.stream):
            rep.stream.wait_event(ready)
            if rep.consumed is not None:
                rep.stream.wait_event(rep.consumed)
            per = inputs[0].shape[0]
            for t, src in enumerate(inputs):
                rep.static_in[t * per : (t + 1) * per].copy_(src)
            rep.graph.replay()
            rep.done = torch.cuda.Event()
            rep.done.record(rep.stream)
        main.wait_event(rep.done)
        self.replays += 1
        return rep.static_out, rep

    @staticmethod
    def release(rep: Any) -> None:
        import torch
        rep.consumed = torch.cuda.Event()
        rep.consumed.record(torch.cuda.current_stream(rep.static_out.device))


class _Race:
    """In-place race of eager vs graph arms over real decoder calls, carried across decodes until decided. An arm is
    timed per run of consecutive calls (replicas overlap, so one call's own interval undercounts them): run time over
    run calls."""

    def __init__(
        self,
        arms: tuple,
        forced: bool = False,
    ) -> None:
        self.arms = (0, *arms)
        self.forced = forced
        self.calls: dict = {a: 0 for a in self.arms}
        self.ms: dict = {a: 0.0 for a in self.arms}
        self.decided: Optional[int] = None
        self.graphs: Optional[TileGraphs] = None
        self.warm = False
        self.reference: Optional[tuple] = None
        self.pending: list = []
        self.prints: Optional[tuple] = None
        self.medians: dict = {}
        self.per_replica = 0

    @staticmethod
    def needed(arm: int) -> int:
        return max(GRAPH_RACE_CALLS, arm)

    def counted(self, arm: int) -> int:
        return self.calls[arm] + sum(1 for a, _s, _e in self.pending if a == arm)

    def skip(self, arm: int) -> None:
        """An arm that cannot run (its pool would pass the cap): never picked."""
        self.calls[arm] = self.needed(arm)
        self.ms[arm] = float("inf")

    def current(self) -> int:
        if self.decided is not None:
            return self.decided
        if self.forced:
            # forced replicas: one eager call first gives the exactness reference, then the graphs
            return self.arms[-1] if self.reference is not None else 0
        for a in self.arms:
            if self.counted(a) < self.needed(a):
                return a
        return 0

    def settle(self) -> None:
        """Read the timed calls (one sync, at the end of a decode) and decide when every arm has its calls."""
        k = 0
        while k < len(self.pending):
            arm, start, _ = self.pending[k]
            j = k
            while j + 1 < len(self.pending) and self.pending[j + 1][0] == arm:
                j += 1
            end = self.pending[j][2]
            end.synchronize()
            self.ms[arm] += start.elapsed_time(end)
            self.calls[arm] += j - k + 1
            k = j + 1
        self.pending = []
        if self.decided is not None:
            return
        if self.graphs is not None and self.graphs.failed is not None:
            self.decided = 0
        elif all(self.calls[a] >= self.needed(a) for a in self.arms):
            per = {a: self.ms[a] / max(1, self.calls[a]) for a in self.arms}
            graph_arms = [a for a in self.arms if a]
            best = min(per[a] for a in graph_arms)
            pick = min(a for a in graph_arms if per[a] <= best * (1 + GRAPH_MARGIN))
            self.decided = pick if per[pick] < per[0] * (1 - GRAPH_MARGIN) else 0
            self.medians = per
        if self.decided is not None and self.graphs is not None:
            if self.decided == 0:
                self.graphs.drop()
            else:
                self.graphs.keep(self.decided)


def graph_pool_cap(device: Any) -> int:
    """Bytes the kept tile graphs may hold: their private pools stay reserved between renders."""
    import torch

    try:
        total = torch.cuda.get_device_properties(device).total_memory
    except Exception:  # noqa: BLE001
        return GRAPH_POOL_MAX_BYTES
    return int(min(GRAPH_POOL_MAX_BYTES, total * GRAPH_POOL_MAX_FRACTION))


def _weights_print(vae: Any) -> tuple:
    """Where the decoder's weights live (a recorded graph reads these addresses)."""
    return tuple(p.data_ptr() for p in vae.decoder.parameters()) + tuple(
        p.data_ptr() for p in vae.post_quant_conv.parameters()
    )


def _stats(vae: Any) -> dict:
    return vae.__dict__.setdefault("_unsloth_canvas_stats", {})


def _global_graphs_off() -> bool:
    from .diffusion_cuda_graph import cuda_graph_disabled
    return cuda_graph_disabled()


def _graph_arms(vae: Any, device: Any) -> tuple:
    from .diffusion_cuda_graph import cuda_graph_disabled

    forced = tile_graphs_setting()
    if forced == 0 or device.type != "cuda" or not vae.__dict__.get("_unsloth_canvas_graphs"):
        return ()
    if cuda_graph_disabled():  # the global CUDA graph kill switches cover tile graphs too
        return ()
    if forced is not None:
        return (1,) if _is_rocm() else (forced,)
    return GRAPH_REPLICAS_ROCM if _is_rocm() else GRAPH_REPLICAS


def canvas_decode(
    vae: Any,
    z: Any,
    decode: Any = None,
) -> Optional[Any]:
    """Tiled decode of ``z`` (4-D image or 5-D clip) into one canvas, or None when the layout is not covered
    (the caller then runs its stock loop). ``decode`` maps a latent tile batch to pixels (default
    ``decoder(post_quant_conv(.))``)."""
    import torch

    plan = layout(vae, z)
    if plan is None:
        return None
    y, x, ratio = plan
    fn = decode or (lambda t: vae.decoder(vae.post_quant_conv(t)))
    tiles = [
        (i, j, z[..., ys // ratio : (ys + yl) // ratio, xs // ratio : (xs + xl) // ratio])
        for i, (ys, yl) in enumerate(zip(y[0], y[1]))
        for j, (xs, xl) in enumerate(zip(x[0], x[1]))
    ]
    equal = len({t[2].shape for t in tiles}) == 1
    batch = tile_batch_size(vae, tiles[0][2], len(tiles)) if equal else 1
    per = int(z.shape[0])
    stats = _stats(vae)
    stats.update(tiles = len(tiles), batch = batch, graph_replicas = 0, graph_replays = 0)
    arms = _graph_arms(vae, z.device) if equal else ()
    race: Optional[_Race] = None
    if arms:
        races = vae.__dict__.setdefault("_unsloth_canvas_races", {})
        key = (str(z.device), tuple(tiles[0][2].shape), batch, str(z.dtype), arms)
        race = races.get(key)
        if race is None:
            if len(races) >= 4:
                races.clear()
            race = races[key] = _Race(arms, forced = tile_graphs_setting() is not None)
        for other in races.values():
            # Graphs of one tile shape at a time: another shape's pools go (its decision stays; a return re-records).
            if other is not race and other.graphs is not None and other.graphs.replicas:
                other.graphs.drop()
        prints = _weights_print(vae)
        if race.prints is None:
            race.prints = prints
        elif race.prints != prints:
            # The weights moved (sequential placement re-uploads the VAE per render): a graph would read the old
            # addresses, and recording per render costs more than it saves, so this load decodes eager.
            vae._unsloth_canvas_graphs = False
            for r in races.values():
                if r.graphs is not None:
                    r.graphs.drop()
            races.clear()
            stats["graph_off"] = "weights moved between decodes"
            race = None
    canvas = None
    ready = None
    start = 0
    calls = 0
    while start < len(tiles):
        group = tiles[start : start + batch]
        inputs = [t[2] for t in group]
        arm = race.current() if race is not None and len(group) == batch else 0
        rep = None
        timed = None
        if race is not None and race.decided is None and not race.forced and len(group) == batch:
            if arm == 0 and not race.warm:
                race.warm = True
            else:
                timed = (torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True))
                timed[0].record()
        try:
            if arm:
                if race.graphs is None:
                    shape = (per * len(group), *inputs[0].shape[1:])
                    race.graphs = TileGraphs(fn, shape, inputs[0].dtype, z.device)
                cap = graph_pool_cap(z.device)
                if race.per_replica and not race.forced and race.per_replica * arm > cap:
                    race.skip(arm)
                    arm, timed = 0, None
            if arm and len(race.graphs.replicas) < arm:
                have = len(race.graphs.replicas)
                sample, reference = race.reference or (None, None)
                if sample is None:
                    sample = torch.cat(inputs) if len(inputs) > 1 else inputs[0]
                grown = race.graphs.grow(arm, sample, reference)
                if grown is None:
                    arm, timed = 0, None  # capture refused for now: eager, untimed, try again later
                elif not grown:
                    stats["graph_failed"] = race.graphs.failed
                    race.decided = 0
                    race.graphs = None
                    arm, timed = 0, None
                elif race.forced:
                    race.decided = arm
                else:
                    race.per_replica = race.per_replica or race.graphs.pool_bytes // arm
                    if race.graphs.pool_bytes > cap:
                        if race.decided is None:
                            for a_ in race.arms:
                                if a_ >= arm:
                                    race.skip(a_)
                            race.graphs.keep(have)
                        else:
                            race.decided = 0
                            race.graphs.drop()
                        arm, timed = 0, None
                    elif timed is not None:
                        timed[0].record()
            if arm:
                if ready is None or ready[0] != arm:
                    ready = (arm, torch.cuda.Event())
                    ready[1].record()
                out, rep = race.graphs.launch(calls, inputs, ready[1])
            else:
                out = fn(torch.cat(inputs) if len(inputs) > 1 else inputs[0])
                if race is not None and race.decided is None and race.reference is None:
                    if len(group) == batch:
                        sample = torch.cat(inputs) if len(inputs) > 1 else inputs[0]
                        race.reference = (sample.clone(), out.clone())
        except Exception as exc:  # noqa: BLE001
            # batching raised the peak, so an OOM with several tiles retries them one at a time
            if len(group) == 1 or not _is_oom(exc):
                raise
            torch.cuda.empty_cache()
            batch = 1
            if race is not None:
                race.pending = []  # runs cut by the OOM would join the next decode's runs
            race = None
            continue
        if canvas is None:
            canvas = Canvas(out[:per], y[0][-1] + y[1][-1], x[0][-1] + x[1][-1], y, x)
        parts = out.split(per) if len(group) > 1 else [out]
        for (i, j, _), part in zip(group, parts):
            canvas.add(i, j, part)
        del parts, out
        if rep is not None:
            TileGraphs.release(rep)
            stats["graph_replays"] += 1
        if timed is not None:
            timed[1].record()
            race.pending.append((arm, *timed))
        start += len(group)
        calls += 1
    stats["calls"] = calls
    if race is not None:
        race.settle()
        if race.decided is not None:
            race.reference = None
        stats["graph_decided"] = race.decided
        stats["graph_replicas"] = len(race.graphs.replicas) if race.graphs is not None else 0
        stats["graph_pool_bytes"] = race.graphs.pool_bytes if race.graphs is not None else 0
        if getattr(race, "medians", None):
            stats["graph_race_ms"] = {str(k): round(v, 3) for k, v in race.medians.items()}
    return canvas.out


def half_decoder_ok(vae: Any) -> bool:
    """A ViT decoder whose blocks keep norms, layer scales and the residual stream fp32 by their own casts (the
    MiniMax decoders: ``norm1(h.float()).to(dtype of attn.to_q)``)."""
    blocks = getattr(getattr(vae, "decoder", None), "transformer_blocks", None)
    if not blocks:
        return False
    b = blocks[0]
    return all(
        hasattr(b, n) for n in ("attn", "ff", "norm1", "norm2", "scale1", "scale2")
    ) and hasattr(b.attn, "to_q")


def _block_linears(vae: Any) -> list:
    import torch

    out = []
    for block in vae.decoder.transformer_blocks:
        for part in (block.attn, block.ff):
            out.extend(m for m in part.modules() if isinstance(m, torch.nn.Linear))
    return out


def set_decoder_dtype(vae: Any, dtype: Any) -> int:
    linears = _block_linears(vae)
    for m in linears:
        m.to(dtype)
    return len(linears)


def _probe(out: Any) -> Optional[str]:
    """``"nonfinite"`` / ``"flat"`` for an output the fp16 decoder may have broken, else None (one host sync)."""
    import torch

    flat = out.detach()
    finite = torch.isfinite(flat).all()
    spread = (flat.amax() - flat.amin()).float()
    code = int(((~finite).int() * 2 + (spread < 1e-4).int()).item())
    if code >= 2:
        return "nonfinite"
    return "flat" if code else None


def _bad_output(out: Any) -> bool:
    return _probe(out) is not None


def _decode_entry(vae: Any, z: Any, stock: Any) -> Any:
    out = canvas_decode(vae, z) if canvas_enabled() else None
    return stock(z) if out is None else out


def _image_decode(self: Any, z: Any) -> Any:
    stock = self.__dict__["_unsloth_canvas_stock_decode"]
    stats = _stats(self)
    stats.clear()
    if not self.use_tiling:
        out = stock(z)
    else:
        out = _decode_entry(self, z, lambda t: stock(t))
    stats["canvas"] = "tiles" in stats
    half = bool(self.__dict__.get("_unsloth_canvas_half"))
    stats["dtype"] = "fp16" if half else "fp32"
    if not half:
        return out
    verdict = _probe(out)
    if verdict is None:
        return out
    # Re-decode in the load dtype (fp32 weights from the fp16 ones; the overflow is in the activations).
    full = self.post_quant_conv.weight.dtype
    for r in (self.__dict__.pop("_unsloth_canvas_races", None) or {}).values():
        if r.graphs is not None:
            r.graphs.drop()
    set_decoder_dtype(self, full)
    self._unsloth_canvas_half = False
    again = _image_decode(self, z)
    if verdict == "flat" and _probe(again) == "flat":
        # A flat image (a solid-colour prompt) decodes flat in fp32 too: fp16 was not the cause, so it stays.
        import torch

        set_decoder_dtype(self, torch.float16)
        self._unsloth_canvas_half = True
        self.__dict__.pop("_unsloth_canvas_races", None)
        _stats(self)["half_flat_check"] = True
        return again
    _stats(self)["half_fallback"] = verdict
    logger = self.__dict__.get("_unsloth_canvas_logger")
    if logger is not None:
        logger.warning(
            "diffusion.vae_canvas: the fp16 decoder gave a %s image; decoding in fp32 for this load",
            verdict,
        )
    return again


def install(
    vae: Any,
    device: Any = None,
    *,
    speed: bool = True,
    half: bool = False,
    logger: Any = None,
) -> tuple:
    """Route an image VAE's tiled ``_decode`` through the canvas; ``speed`` adds tile batching and the graph race
    on CUDA, ``half`` the fp16 decoder (callers opt in per family). Idempotent; returns the engaged levers."""
    import torch

    if vae is None or not stock_tiling(vae) or not callable(getattr(vae, "_decode", None)):
        return ()
    if hasattr(type(vae), "_decode_clip"):
        return ()  # video VAEs composite inside their own clip loop
    dev = torch.device(device) if device is not None else None
    levers = []
    with _LOCK:
        if "_unsloth_canvas_stock_decode" not in vae.__dict__:
            vae._unsloth_canvas_stock_decode = vae._decode
            vae._decode = types.MethodType(_image_decode, vae)
        vae._unsloth_canvas_logger = logger
        cuda = dev is not None and dev.type == "cuda" and torch.cuda.is_available()
        vae._unsloth_canvas_graphs = bool(speed and cuda)
        # Speed off keeps one tile per call (stock memory and numerics); the canvas seam fix still applies.
        vae._unsloth_canvas_batch_ok = bool(speed and cuda)
        if canvas_enabled():
            levers.append(LEVER_CANVAS)
        if speed and cuda and tile_batch_setting() != 1:
            levers.append(LEVER_TILE_BATCH)
        if vae._unsloth_canvas_graphs and tile_graphs_setting() != 0 and not _global_graphs_off():
            levers.append(LEVER_TILE_GRAPHS)
        want_half = half and speed and cuda and half_enabled() and half_decoder_ok(vae)
        if want_half and not vae.__dict__.get("_unsloth_canvas_half"):
            set_decoder_dtype(vae, torch.float16)
            vae._unsloth_canvas_half = True
        if vae.__dict__.get("_unsloth_canvas_half"):
            levers.append(LEVER_HALF)
    if logger is not None and levers:
        logger.info(
            "diffusion.vae_canvas: %s decodes with %s", type(vae).__name__, ",".join(levers)
        )
    return tuple(levers)


def release_graphs(vae: Any) -> None:
    """Drop the tile graphs and their private pools (an unload: they would outlive the freed weights for as long
    as anything still references the VAE)."""
    for r in (vae.__dict__.pop("_unsloth_canvas_races", None) or {}).values():
        if r.graphs is not None:
            r.graphs.drop()
            r.graphs = None
        r.reference = None


def uninstall(vae: Any) -> None:
    stock = vae.__dict__.pop("_unsloth_canvas_stock_decode", None)
    if stock is not None:
        vae.__dict__.pop("_decode", None)
    if vae.__dict__.pop("_unsloth_canvas_half", False):
        set_decoder_dtype(vae, vae.post_quant_conv.weight.dtype)
    release_graphs(vae)
    for name in (
        "_unsloth_canvas_graphs",
        "_unsloth_canvas_logger",
        "_unsloth_canvas_stats",
        "_unsloth_canvas_batches",
    ):
        vae.__dict__.pop(name, None)
