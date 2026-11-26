# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Measured verdicts on whether concurrent image requests should share one call on this load, card and shape.

A race times k requests served one at a time against the same k requests in one call (each arm including encode,
denoise and decode), interleaved over several repeats, on an idle engine only. A batch size is promoted only when the
lower one-sided 95% bound of the per-repeat speedup is at least ``PROMOTE_LB`` and the batched p95 request latency is at
most ``P95_SLACK`` times the serial one. Winners and losers are both persisted, keyed by the probe fingerprint (torch,
driver, card, env) plus the load's model, variant, precision, attention, placement, graph and compile state, canvas,
guidance and batch size. A (load, shape, size) without a verdict keeps today's rules."""

from __future__ import annotations

import math
import random
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, Sequence

PROMOTE_LB = 1.05
P95_SLACK = 1.03
MIN_REPEATS = 3
RACE_SIZES = (2, 4)
_N_BOOT = 4000
# One-sided 95% Student t quantiles by degrees of freedom (a df between rows uses the smaller df: more conservative).
_T95 = (
    (1, 6.314),
    (2, 2.920),
    (3, 2.353),
    (4, 2.132),
    (5, 2.015),
    (6, 1.943),
    (7, 1.895),
    (8, 1.860),
    (9, 1.833),
    (10, 1.812),
    (12, 1.782),
    (15, 1.753),
    (20, 1.725),
    (30, 1.697),
    (60, 1.671),
)

# Generic prompts of similar length, so a T5-bucket family keys them alike.
RACE_PROMPTS = (
    "a red fox sitting in fresh snow, photograph",
    "a blue sports car on a desert highway at noon",
    "a bowl of ramen with a soft boiled egg, top view",
    "a lighthouse on a rocky cliff at sunset, painting",
    "a green parrot on a branch in the rain forest",
    "a wooden cabin beside a frozen lake at night",
    "a bicycle leaning on a brick wall in the city",
    "a cup of coffee on a table by a sunny window",
)
NEIGHBOUR_PROMPTS = (
    "an old steam train crossing a stone bridge",
    "a field of sunflowers under a stormy sky",
    "a robot reading a newspaper in a diner",
    "a jellyfish glowing in a dark blue ocean",
    "a stack of pancakes with berries and syrup",
    "a snowy mountain peak above the clouds",
    "a cat sleeping on a pile of old books",
)

_STORE: list = []
# Cards with no fingerprint (no driver identity, no CUDA): status polls must not re-probe them every call.
_UNIDENTIFIED: dict = {}
_UNIDENTIFIED_RETRY_S = 300.0


def _store() -> Any:
    if not _STORE:
        from .diffusion_probe_cache import Namespace
        _STORE.append(
            Namespace(
                "batch_race",
                sources = (
                    "diffusion_batch_race.py",
                    "diffusion_batch_adapters.py",
                    "diffusion_batched.py",
                    "diffusion_cuda_graph.py",
                ),
            )
        )
    return _STORE[0]


def t95(df: int) -> float:
    if df < 1:
        return float("inf")
    best = _T95[0][1]
    for row_df, value in _T95:
        if row_df <= df:
            best = value
    return best


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100)."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return float(ordered[min(rank, len(ordered)) - 1])


def speedup_lower_bound(ratios: Sequence[float], *, seed: int = 0) -> float:
    """One-sided 95% lower bound of the geometric-mean speedup: the smaller of a Student t bound on the log ratios
    and a seeded percentile bootstrap bound (the bootstrap under-covers at small n). 0 when fewer than two."""
    logs = [math.log(r) for r in ratios if r and r > 0 and math.isfinite(r)]
    n = len(logs)
    if n < 2:
        return 0.0
    mean = sum(logs) / n
    var = sum((x - mean) ** 2 for x in logs) / (n - 1)
    t_bound = mean - t95(n - 1) * math.sqrt(var / n)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(logs) for _ in range(n)) / n for _ in range(_N_BOOT))
    boot_bound = means[int(0.05 * _N_BOOT)]
    return math.exp(min(t_bound, boot_bound))


def burst_latencies(serial_times: Sequence[float], batched_time: float) -> tuple[list, list]:
    """Request latencies of one burst of k: served one at a time (each waits for those before it) vs in one call."""
    out, total = [], 0.0
    for t in serial_times:
        total += t
        out.append(total)
    return out, [float(batched_time)] * len(serial_times)


def judge(repeats: Sequence[tuple[Sequence[float], float]], *, seed: int = 0) -> dict:
    """``repeats``: per repeat (the k serial render times, the one batched call time). The verdict for this size."""
    ratios, serial_lat, batched_lat = [], [], []
    for serial, batched in repeats:
        if not serial or batched <= 0:
            continue
        ratios.append(sum(serial) / batched)
        s, b = burst_latencies(serial, batched)
        serial_lat.extend(s)
        batched_lat.extend(b)
    lb = speedup_lower_bound(ratios, seed = seed)
    p95_s, p95_b = percentile(serial_lat, 95), percentile(batched_lat, 95)
    p50_s, p50_b = percentile(serial_lat, 50), percentile(batched_lat, 50)
    enough = len(ratios) >= MIN_REPEATS
    fast = lb >= PROMOTE_LB
    tail = bool(serial_lat) and p95_b <= P95_SLACK * p95_s
    if not enough:
        reason = f"{len(ratios)} repeats, fewer than {MIN_REPEATS}"
    elif not fast:
        reason = f"speedup lower bound {lb:.3f} below {PROMOTE_LB}"
    elif not tail:
        reason = f"batched p95 {p95_b:.2f}s above {P95_SLACK} x serial p95 {p95_s:.2f}s"
    else:
        reason = f"speedup lower bound {lb:.3f}, p95 {p95_b:.2f}s vs {p95_s:.2f}s"
    ordered = sorted(ratios)
    return {
        "promoted": bool(enough and fast and tail),
        "reason": reason,
        "repeats": len(ratios),
        "speedup_lb": round(lb, 4),
        "speedup_median": round(ordered[len(ordered) // 2], 4) if ordered else None,
        "ratios": [round(r, 4) for r in ratios],
        "p50_serial_s": round(p50_s, 3) if serial_lat else None,
        "p50_batched_s": round(p50_b, 3) if serial_lat else None,
        "p95_serial_s": round(p95_s, 3) if serial_lat else None,
        "p95_batched_s": round(p95_b, 3) if serial_lat else None,
    }


# ------------------------------------------------------------------------------------------------ keys


def _compile_state(state: Any) -> str:
    optims = tuple(getattr(state, "speed_optims", ()) or ())
    if "compiled" not in optims:
        return "eager"
    try:
        from .diffusion import compiled_shapes_are_static
        static = compiled_shapes_are_static(
            getattr(state, "pipe", None), getattr(state, "speed_mode", None)
        )
    except Exception:  # noqa: BLE001
        static = True
    return "static" if static else "dynamic"


def topology(state: Any) -> str:
    """Cards the host shows and how this load spans them (component split, CFG replica, layer split): a verdict on one
    topology says nothing about another (Kaggle 2xT4: per-card replicas beat the split only from 2 users up)."""
    try:
        import torch
        count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        count = 0
    pipe = getattr(state, "pipe", None)
    parts = [f"cards{count}"]
    try:
        from . import diffusion_multigpu as mg

        info = mg._PLACEMENTS.get(pipe) if pipe is not None else None
        if info:
            moved = ",".join(sorted(str(m) for m in info.get("moved") or ()))
            parts.append(f"split:{info.get('primary')}>{info.get('secondary')}:{moved}")
        if pipe is not None and mg._CFG_ROUTERS.get(pipe) is not None:
            parts.append("cfg_replica")
        from .diffusion_layer_split_plan import layer_split_status

        if pipe is not None and layer_split_status(pipe):
            parts.append("layer_split")
    except Exception:  # noqa: BLE001 - unreadable placement: its own key, never another load's verdict
        parts.append("placement?")
    return "+".join(parts)


def load_key(state: Any) -> str:
    """Everything about the load a verdict depends on (the card side comes from the probe fingerprint)."""
    from .diffusion import _denoiser_streams_from_host

    family = getattr(state, "family", None)
    pipe = getattr(state, "pipe", None)
    resolved = getattr(state, "resolved", None) or {}
    revision = ""
    if isinstance(resolved, dict):
        revision = str(
            resolved.get("revision") or resolved.get("sha") or resolved.get("commit") or ""
        )
    try:
        streamed = bool(_denoiser_streams_from_host(state))
    except Exception:  # noqa: BLE001
        streamed = True
    parts = (
        str(getattr(state, "repo_id", None) or ""),
        revision,
        str(getattr(state, "gguf_filename", None) or ""),
        str(getattr(family, "name", None) or ""),
        type(getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)).__name__,
        str(getattr(state, "transformer_quant", None) or "bf16"),
        str(getattr(state, "dtype", None) or ""),
        str(getattr(state, "speed_mode", None) or ""),
        ",".join(sorted(str(o) for o in (getattr(state, "speed_optims", ()) or ()))),
        str(getattr(state, "attention_backend", None) or ""),
        str(getattr(state, "offload_policy", None) or ""),
        "streamed" if streamed else "resident",
        "graphs" if getattr(state, "cuda_graphs", ()) else "nographs",
        _compile_state(state),
        str(getattr(state, "transformer_cache", None) or ""),
        topology(state),
    )
    return "|".join(parts)


def shape_key(
    width: Any,
    height: Any,
    guidance: Any,
    steps: Any = None,
) -> str:
    """Canvas, guidance and step count: the fixed per-call cost (encode, decode) weighs more at few steps, so a win at
    one step count says little about another (one engine's own FLUX.3 batching: 2.00x at 512 / 8 steps, 1.03x at
    1024 / 50)."""
    try:
        g = "" if guidance is None else f"{float(guidance):.3g}"
    except (TypeError, ValueError):
        g = str(guidance)
    try:
        n = "" if steps is None else str(int(steps))
    except (TypeError, ValueError):
        n = str(steps)
    return f"{int(width or 0)}x{int(height or 0)}|g{g}|s{n}|default"


def card_of(state: Any) -> Optional[str]:
    device = str(getattr(state, "device", "") or "")
    if not device.startswith("cuda"):
        return None
    ordinal = getattr(state, "placed_ordinal", None)
    if ordinal is None:
        ordinal = getattr(state, "gpu_ordinal", None)
    card = f"cuda:{int(ordinal)}" if ordinal is not None else device
    seen = _UNIDENTIFIED.get(card)
    if seen is not None and time.monotonic() - seen < _UNIDENTIFIED_RETRY_S:
        return None
    return card


def _slot(card: str) -> Optional[tuple]:
    slot = _store()._slot(card)
    if slot is None:
        _UNIDENTIFIED[card] = time.monotonic()
    else:
        _UNIDENTIFIED.pop(card, None)
    return slot


def verdict_key(
    state: Any,
    width: Any,
    height: Any,
    guidance: Any,
    size: int,
    steps: Any = None,
) -> str:
    return f"{load_key(state)}#{shape_key(width, height, guidance, steps)}#k{int(size)}"


def lookup(
    state: Any,
    width: Any,
    height: Any,
    guidance: Any,
    sizes: Sequence[int],
    steps: Any = None,
) -> dict[int, dict]:
    """Persisted verdicts for this load and shape by batch size; {} when there are none or no card identity."""
    card = card_of(state)
    if card is None:
        return {}
    try:
        slot = _slot(card)
        if slot is None:
            return {}
        base = f"{load_key(state)}#{shape_key(width, height, guidance, steps)}#k"
        out = {}
        for size in sizes:
            found = slot[2].get(f"{base}{int(size)}")
            if isinstance(found, dict):
                out[int(size)] = found
        return out
    except Exception:  # noqa: BLE001 - an unreadable store is no verdict
        return {}


def any_promoted(state: Any) -> bool:
    """Whether some shape of this load has a promoted batch size (status reports the measured mode)."""
    card = card_of(state)
    if card is None:
        return False
    try:
        slot = _slot(card)
        if slot is None:
            return False
        prefix = f"{load_key(state)}#"
        return any(
            k.startswith(prefix) and isinstance(v, dict) and v.get("promoted")
            for k, v in slot[2].items()
        )
    except Exception:  # noqa: BLE001
        return False


def record(
    state: Any,
    width: Any,
    height: Any,
    guidance: Any,
    size: int,
    verdict: dict,
    steps: Any = None,
) -> bool:
    card = card_of(state)
    if card is None:
        return False
    try:
        return bool(
            _store().put(
                card, verdict_key(state, width, height, guidance, size, steps), dict(verdict)
            )
        )
    except Exception:  # noqa: BLE001 - a cache write never fails the race
        return False


# ------------------------------------------------------------------------------------------------ runner


class RaceRefused(RuntimeError):
    """The engine is not idle (or nothing is loaded): no race runs on a user's request."""


def _pixels(images: Sequence[Any]) -> list:
    out = []
    for image in images:
        try:
            import numpy as np
            out.append(np.asarray(image.convert("RGB")).copy())
        except Exception:  # noqa: BLE001
            out.append(None)
    return out


def _same(a: Any, b: Any) -> Optional[bool]:
    if a is None or b is None:
        return None
    try:
        return bool(a.shape == b.shape and (a == b).all())
    except Exception:  # noqa: BLE001
        return None


def _max_abs(a: Any, b: Any) -> Optional[int]:
    if a is None or b is None or a.shape != b.shape:
        return None
    return int(abs(a.astype("int16") - b.astype("int16")).max())


@contextmanager
def _graphs_bypassed(handles: Sequence[Any]) -> Iterator[None]:
    """The batched-eager arm: every graph handle runs eager (``set_bypass``) and keeps its graphs."""
    flipped = []
    try:
        for handle in handles:
            if callable(getattr(handle, "set_bypass", None)):
                handle.set_bypass(True)
                flipped.append(handle)
        yield
    finally:
        for handle in flipped:
            try:
                handle.set_bypass(False)
            except Exception:  # noqa: BLE001
                pass


def run_race(
    generate: Callable[..., dict],
    state: Any,
    *,
    idle: Callable[[], bool],
    width: int,
    height: int,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
    sizes: Sequence[int] = RACE_SIZES,
    repeats: int = 5,
    accuracy: bool = True,
    allowed_sizes: Optional[Callable[[int], bool]] = None,
    seed: int = 0,
    persist: bool = True,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Race serial against batched at each size on an idle engine; persist each size's verdict (losers too).

    ``generate`` is the backend's generate. ``idle`` is checked before every render: a request that starts waiting
    aborts the race (nothing is persisted for the size under way). ``allowed_sizes`` (static compile) skips a size
    that would compile."""
    from . import diffusion_cuda_graph as cuda_graph

    if not idle():
        raise RaceRefused("a request holds or waits for the generation slot")
    rng = random.Random(seed)
    common = {"width": int(width), "height": int(height), "live_preview": False}
    if steps is not None:
        common["steps"] = int(steps)
    if guidance is not None:
        common["guidance"] = float(guidance)
    handles = tuple(getattr(state, "cuda_graphs", ()) or ())
    report: dict = {"width": int(width), "height": int(height), "guidance": guidance, "sizes": {}}

    class _Aborted(Exception):
        pass

    def solo(prompt: str, seed_: int) -> tuple[float, Any]:
        if not idle():
            raise _Aborted()
        start = clock()
        out = generate(prompt = prompt, seed = seed_, **common)
        return clock() - start, (out.get("images") or [None])[0]

    def batch(
        prompts: list,
        seeds: list,
        *,
        eager: bool = False,
        capture: bool = False,
    ) -> tuple[float, list]:
        if not idle():
            raise _Aborted()
        start = clock()
        with (
            cuda_graph.batch_rows(len(prompts), capture = capture),
            _graphs_bypassed(handles) if eager else _nullctx(),
        ):
            out = generate(
                prompt = prompts[0],
                prompts = list(prompts),
                seeds = list(seeds),
                batch_size = len(prompts),
                **common,
            )
        return clock() - start, list(out.get("images") or [])

    # Largest first: inside one pool the smaller graphs then reuse the blocks the largest recorded.
    for size in sorted({int(k) for k in sizes}, reverse = True):
        if size < 2:
            continue
        if allowed_sizes is not None and not allowed_sizes(size):
            report["sizes"][str(size)] = {
                "skipped": "static compile has no graph for this batch size"
            }
            continue
        prompts = list(RACE_PROMPTS[:size])
        seeds = [1000 + 7 * i for i in range(size)]
        arms = ["batched"] + (["batched_eager"] if handles else [])
        try:
            # Warm-ups (untimed): the serial shape, then each batched arm; a batched graph at this size records here.
            solo(prompts[0], seeds[0])
            batch(prompts, seeds, capture = True)
            if "batched_eager" in arms:
                batch(prompts, seeds, eager = True)
            samples: dict[str, list] = {arm: [] for arm in arms}
            for _ in range(max(1, int(repeats))):
                order = ["serial", *arms]
                rng.shuffle(order)
                serial_times: list = []
                times: dict = {}
                for arm in order:
                    if arm == "serial":
                        serial_times = [solo(p, s)[0] for p, s in zip(prompts, seeds)]
                    else:
                        times[arm] = batch(
                            prompts, seeds, eager = arm == "batched_eager", capture = True
                        )[0]
                for arm in arms:
                    samples[arm].append((serial_times, times[arm]))
            verdicts = {arm: judge(samples[arm], seed = seed) for arm in arms}
            best = max(arms, key = lambda arm: verdicts[arm]["speedup_lb"])
            verdict = dict(verdicts[best], arm = best, arms = verdicts, size = size, t = time.time())
            if accuracy:
                verdict.update(
                    _neighbour_check(solo, batch, prompts, seeds, eager = best == "batched_eager")
                )
        except _Aborted:
            report["sizes"][str(size)] = {"aborted": "a request started waiting for the slot"}
            report["aborted"] = True
            break
        report["sizes"][str(size)] = verdict
        if not verdict.get("promoted"):
            # A losing size never runs batched here: its graphs would only pin pool memory.
            dropped = 0
            for handle in handles:
                drop = getattr(handle, "drop_batched", None)
                if callable(drop):
                    try:
                        dropped += int(drop(size) or 0)
                    except Exception:  # noqa: BLE001
                        pass
            verdict["batched_graphs_dropped"] = dropped
        if persist:
            verdict["persisted"] = record(state, width, height, guidance, size, verdict, steps)
    return report


@contextmanager
def _nullctx() -> Iterator[None]:
    yield


def _neighbour_check(solo, batch, prompts: list, seeds: list, *, eager: bool) -> dict:
    """G2a at this batch shape: change only the neighbours (prompt and seed), the target first then last; an A/A
    repeat of the identical batch gives the noise floor. Plus whether a member equals its solo render (replay)."""
    size = len(prompts)
    others = list(NEIGHBOUR_PROMPTS[: size - 1])
    other_seeds = [9000 + 13 * i for i in range(size - 1)]
    _, a = batch(prompts, seeds, eager = eager)
    _, a2 = batch(prompts, seeds, eager = eager)
    _, b = batch([prompts[0], *others], [seeds[0], *other_seeds], eager = eager)
    _, c = batch([*others, prompts[0]], [*other_seeds, seeds[0]], eager = eager)
    _, d = batch([*prompts[1:], prompts[0]], [*seeds[1:], seeds[0]], eager = eager)
    _, alone = solo(prompts[0], seeds[0])
    pa, pa2, pb, pc, pd = (_pixels(x) for x in (a, a2, b, c, d))
    [ps] = _pixels([alone])
    first = _same(pa[0], pb[0]) if pa and pb else None
    last = _same(pd[-1], pc[-1]) if pd and pc else None
    aa = _same(pa[0], pa2[0]) if pa and pa2 else None
    return {
        "aa_exact": aa,
        "neighbour_exact": bool(first and last),
        "neighbour_first_max_abs": _max_abs(pa[0], pb[0]) if pa and pb else None,
        "neighbour_last_max_abs": _max_abs(pd[-1], pc[-1]) if pd and pc else None,
        "aa_max_abs": _max_abs(pa[0], pa2[0]) if pa and pa2 else None,
        "replay_exact": bool(_same(pa[0], ps)) if pa else False,
        "replay_max_abs": _max_abs(pa[0], ps) if pa else None,
    }


def seeded_join_ok(verdict: Optional[dict]) -> bool:
    """A seeded request may join a batch of this size only when the measured path is neighbour-exact (G2a) and its
    row replays bit for bit from the seed alone (G2b): otherwise the returned seed would not reproduce the image."""
    return bool(
        isinstance(verdict, dict)
        and verdict.get("promoted")
        and verdict.get("neighbour_exact")
        and verdict.get("replay_exact")
    )
