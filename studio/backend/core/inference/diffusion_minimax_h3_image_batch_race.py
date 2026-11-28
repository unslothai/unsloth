# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image: measured verdicts on whether a k-image call runs its rows as one batched forward.

The request-batching race (diffusion_batch_race) times k requests served one at a time against one shared call. This
family never shares a call across requests, and its own k-image call already denoises in lockstep on a streamed DiT,
so that race times the wrong pair. This one times ONE k-image call two ways on an idle engine, interleaved over the
repeats: the batched forward forced off (lockstep on a streamed or sequential DiT, one image after another on a
resident one) against forced on, and judges them by the same rule (diffusion_batch_race.judge: lower one-sided 95%
bound of the speedup at least 1.05x, p95 at most +3%). The batched arm must really run its rows as one call
(``stats["batched"] == [k]``), else the size is skipped, not judged.

Quality: batched rows are bit-identical to their solo renders by construction (no block op mixes rows; the fp32
patch projection and output head run per row), so the offline G2c verdict (diffusion_batch_quality: paired LPIPS of
batched against solo) is met with equality wherever rows are exact. The race checks that on this card and load
instead of trusting it (G1): every batched row equals the same row rendered with the batched forward off, and the
first and last rows equal their solo renders, byte for byte. A size serves only when the speed rule AND that check
pass; ``accuracy=False`` measures speed but promotes nothing. Verdicts persist per probe fingerprint (torch, driver,
card) plus the load key, output size, steps and k, in their own namespace (a change to these sources drops them).

``UNSLOTH_H3_IMAGE_BATCHED`` set to 1 or 0 overrides every verdict."""

from __future__ import annotations

import random
import time
import weakref
from typing import Any, Callable, Optional, Sequence

from . import diffusion_minimax_h3_image_batch as h3_batch

RACE_SIZES = (2, 4)
_STORE: list = []
_BACKENDS: "weakref.WeakSet" = weakref.WeakSet()


def _store() -> Any:
    if not _STORE:
        from .diffusion_probe_cache import Namespace
        _STORE.append(
            Namespace(
                "h3_image_batch",
                sources = (
                    "diffusion_minimax_h3_image_batch_race.py",
                    "diffusion_minimax_h3_image_batch.py",
                    "diffusion_minimax_h3_image.py",
                    "diffusion_minimax_h3_image_speed.py",
                    "video_minimax_h3_rowmod.py",
                ),
            )
        )
    return _STORE[0]


def register_backend(backend: Any) -> None:
    """A backend whose loaded state a pipe call may look its verdicts up in."""
    try:
        _BACKENDS.add(backend)
    except TypeError:
        pass


def state_of(pipe: Any) -> Optional[Any]:
    """The loaded state that holds ``pipe``, None when no registered backend has it loaded."""
    for backend in list(_BACKENDS):
        state = getattr(backend, "_state", None)
        if state is not None and getattr(state, "pipe", None) is pipe:
            return state
    return None


def shape_key(width: Any, height: Any, steps: Any) -> str:
    return f"{int(width)}x{int(height)}|s{int(steps)}"


def verdict_key(state: Any, width: Any, height: Any, steps: Any, size: int) -> str:
    from .diffusion_batch_race import load_key
    return f"{load_key(state)}#{shape_key(width, height, steps)}#k{int(size)}"


def lookup(state: Any, width: Any, height: Any, steps: Any, size: int) -> Optional[dict]:
    """The persisted verdict for this load, card, shape and k; None without one or without a card identity."""
    from .diffusion_batch_race import card_of

    card = card_of(state)
    if card is None:
        return None
    try:
        found = _store().get(card, verdict_key(state, width, height, steps, size))
    except Exception:  # noqa: BLE001 - an unreadable store is no verdict
        return None
    return found if isinstance(found, dict) else None


def record(state: Any, width: Any, height: Any, steps: Any, size: int, verdict: dict) -> bool:
    from .diffusion_batch_race import card_of

    card = card_of(state)
    if card is None:
        return False
    try:
        return bool(
            _store().put(card, verdict_key(state, width, height, steps, size), dict(verdict))
        )
    except Exception:  # noqa: BLE001 - a cache write never fails the race
        return False


def serves(verdict: Any) -> bool:
    """A verdict that lets Auto run the batched forward: the speed rule passed and every row was exact."""
    return bool(
        isinstance(verdict, dict)
        and verdict.get("promoted")
        and verdict.get("engaged")
        and verdict.get("rows_exact")
        and verdict.get("solo_exact")
    )


def measured_size(
    pipe: Any,
    width: Any,
    height: Any,
    steps: Any,
    images: int,
    canvas: Any = None,
) -> Optional[int]:
    """``images`` when a serving verdict exists for a call of that many images at this shape on this load and card
    (and, given ``canvas``, the canvas it was raced at); else None."""
    if images < 2:
        return None
    state = state_of(pipe)
    if state is None:
        return None
    try:
        verdict = lookup(state, width, height, steps, images)
    except Exception:  # noqa: BLE001
        return None
    if not serves(verdict):
        return None
    if canvas is not None and verdict.get("canvas") is not None:
        if [int(x) for x in verdict["canvas"]] != [int(x) for x in canvas]:
            return None
    return int(images)


def _pixels(images: Sequence[Any]) -> list:
    from .diffusion_batch_race import _pixels as pixels
    return pixels(images)


def _all_same(a: list, b: list) -> bool:
    from .diffusion_batch_race import _same
    return len(a) == len(b) and bool(a) and all(_same(x, y) for x, y in zip(a, b))


def run_race(
    generate: Callable[..., dict],
    state: Any,
    *,
    idle: Callable[[], bool],
    width: int,
    height: int,
    steps: int,
    guidance: Optional[float] = None,
    sizes: Sequence[int] = RACE_SIZES,
    repeats: int = 5,
    accuracy: bool = True,
    seed: int = 0,
    persist: bool = True,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Race the batched forward off against on, one k-image call each, at every size; persist each size's verdict
    (losers too). ``idle`` is checked before every render: a request that starts waiting aborts the race."""
    from . import diffusion_batch_race as race
    from .diffusion_minimax_h3_image import H3_IMAGE_GUIDANCE

    bad = race.invalid_race_sizes(list(sizes))
    if bad:
        raise ValueError(bad)
    if not idle():
        raise race.RaceRefused("a request holds or waits for the generation slot")
    pipe = getattr(state, "pipe", None)
    rng = random.Random(seed)
    prompt = race.RACE_PROMPTS[0]
    common = {
        "width": int(width),
        "height": int(height),
        "steps": int(steps),
        # guidance-distilled: generate()'s own default (0) is refused, so the race sends the one value it takes
        "guidance": float(H3_IMAGE_GUIDANCE if guidance is None else guidance),
        "live_preview": False,
    }
    configuration = race._execution_configuration(state)
    report: dict = {
        "kind": "h3_image_batched_forward",
        "arms": [h3_batch.MODE_OFF, h3_batch.MODE_ON],
        "width": int(width),
        "height": int(height),
        "steps": int(steps),
        "sizes": {},
    }

    class _Aborted(Exception):
        pass

    def render(arm: str, seeds: list) -> tuple[float, list, dict]:
        if not idle():
            raise _Aborted()
        with h3_batch.forced_arm(arm):
            start = clock()
            out = generate(prompt = prompt, seeds = list(seeds), batch_size = len(seeds), **common)
            took = clock() - start
        stats = dict(getattr(pipe, "_unsloth_h3_last_stats", None) or {})
        stats["refusal"] = getattr(pipe, "_unsloth_h3_together_refusal", None)
        return took, list(out.get("images") or []), stats

    for size in sorted({int(k) for k in sizes}, reverse = True):
        seeds = [1000 + 7 * i for i in range(size)]
        verdict: dict
        try:
            # Warm-ups (untimed): each arm once; the batched one also says whether its rows really share a call.
            render(h3_batch.MODE_OFF, seeds)
            _, _, warm = render(h3_batch.MODE_ON, seeds)
            batched = list(warm.get("batched") or [])
            if batched != [size]:
                report["sizes"][str(size)] = {
                    "promoted": False,
                    "engaged": False,
                    "skipped": f"the batched forward did not run all {size} rows as one call here "
                    f"(batched {batched or 'off'}; {warm.get('refusal') or 'no refusal'}): nothing to judge",
                    "size": size,
                }
                continue
            samples: list = []
            last: dict = {}
            for _ in range(max(1, int(repeats))):
                order = [h3_batch.MODE_OFF, h3_batch.MODE_ON]
                rng.shuffle(order)
                times: dict = {}
                for arm in order:
                    times[arm], last[arm], stats = render(arm, seeds)
                    if arm == h3_batch.MODE_ON and list(stats.get("batched") or []) != [size]:
                        raise RuntimeError(
                            f"the batched arm stopped running {size} rows as one call "
                            f"({stats.get('batched')})"
                        )
                samples.append(([times[h3_batch.MODE_OFF]], times[h3_batch.MODE_ON]))
            judged = race.judge(samples, seed = seed)
            verdict = dict(
                judged,
                engaged = True,
                size = size,
                canvas = list(warm.get("canvas") or ()) or None,
                placement = warm.get("placement"),
                t = time.time(),
            )
            exact: dict = {"rows_exact": False, "solo_exact": False}
            if accuracy:
                on, off = _pixels(last[h3_batch.MODE_ON]), _pixels(last[h3_batch.MODE_OFF])
                solo = []
                for index in sorted({0, size - 1}):
                    solo.append((index, _pixels(render(h3_batch.MODE_OFF, [seeds[index]])[1])))
                exact = {
                    "rows_exact": _all_same(on, off),
                    "solo_exact": all(
                        len(px) == 1 and _all_same(px, [on[i]]) and _all_same(px, [off[i]])
                        for i, px in solo
                    ),
                    "rows_max_abs": max(
                        (race._max_abs(a, b) or 0 for a, b in zip(on, off)), default = None
                    ),
                }
            verdict.update(exact)
            if verdict["promoted"] and not accuracy:
                verdict["promoted"] = False
                verdict["reason"] += "; no exactness check (accuracy off): not promoted"
            elif verdict["promoted"] and not (exact["rows_exact"] and exact["solo_exact"]):
                verdict["promoted"] = False
                verdict["reason"] += "; batched rows are not bit-identical to their solo renders"
        except _Aborted:
            report["sizes"][str(size)] = {"aborted": "a request started waiting for the slot"}
            report["aborted"] = True
            break
        except Exception as exc:  # noqa: BLE001 - only a capacity failure is a verdict; anything else raises
            if not race._capacity_failure(exc):
                raise
            reason = f"out of memory at batch {size}: {str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__}"
            exc.__traceback__ = None
            del exc
            race._release_cached()
            verdict = {
                "promoted": False,
                "engaged": False,
                "failed": reason,
                "size": size,
                "t": time.time(),
            }
        if race._execution_configuration(state) != configuration:
            raise race.RaceRefused(
                "the load's execution configuration changed during the race; race it again"
            )
        verdict["serves"] = serves(verdict)
        if persist:
            verdict["persisted"] = record(state, width, height, steps, size, verdict)
        report["sizes"][str(size)] = verdict
    return report
