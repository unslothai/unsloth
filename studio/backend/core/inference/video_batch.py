# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Several queued clips denoised as one batch: which loads and requests may share a denoise, the
members that ride along with the clip that started it, and the per-clip VAE decode.

The fair queue stays the only order. When the clip at its head starts and the clips right behind
it ask for the same render (same canvas, frames, steps, guidance and shift, text-to-video only),
the engine takes them as riders of that one denoise: one prompt, negative prompt and seeded
``torch.Generator`` per row, so row i draws the noise its own solo render draws. Each clip is still
decoded, encoded and saved on its own, and a Stop settles its own clip at once. A clip with nobody
waiting behind it starts exactly as before, with no wait.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

ENV = "UNSLOTH_VIDEO_BATCH"
MAX_ENV = "UNSLOTH_VIDEO_BATCH_MAX"
DEFAULT_MAX_ROWS = 4
# How long a started clip waits for the queue to hand over its riders; the drainer seals at once.
SEAL_TIMEOUT_S = 10.0
DECODE_LAYER = "video_batch_per_row"

# Families whose Diffusers pipeline denoises any batch with one generator per row and no per-row state
# outside the latents: the Wan text-to-video pipelines (5B TI2V, A14B two-expert, 2.1).
BATCH_FAMILIES = frozenset(
    {
        "wan2.2-ti2v-5b",
        "wan2.2-t2v-a14b",
        "wan2.1-t2v-1.3b",
        "wan2.1-t2v-14b",
    }
)

_OFF = frozenset({"0", "off", "false", "no", "disable", "disabled"})
_FORCE = frozenset({"force", "always", "on", "1", "true", "yes"})

STATS = {"batches": 0, "rows": 0, "declined": 0}
_STATS_LOCK = threading.Lock()


def mode() -> str:
    """``auto`` (batch where measured to win: a streamed denoiser), ``force`` or ``off``."""
    raw = (os.environ.get(ENV) or "").strip().lower()
    if raw in _OFF:
        return "off"
    if raw in _FORCE:
        return "force"
    return "auto"


def max_rows() -> int:
    try:
        value = int((os.environ.get(MAX_ENV) or "").strip() or DEFAULT_MAX_ROWS)
    except ValueError:
        value = DEFAULT_MAX_ROWS
    return max(1, min(16, value))


def note(stat: str, count: int = 1) -> None:
    with _STATS_LOCK:
        STATS[stat] = STATS.get(stat, 0) + count


def counts() -> dict:
    with _STATS_LOCK:
        return dict(STATS)


def _denoisers(pipe: Any) -> list:
    return [
        module
        for module in (getattr(pipe, "transformer", None), getattr(pipe, "transformer_2", None))
        if module is not None
    ]


def load_decline_reason(state: Any) -> Optional[str]:
    """Why the resident load never batches clips, else None. Every reason keeps today's serial path."""
    chosen = mode()
    if chosen == "off":
        return f"{ENV}=off"
    fam = getattr(state, "family", None)
    if getattr(state, "engine", "diffusers") != "diffusers":
        return "only the Diffusers engine batches clips"
    if getattr(fam, "name", None) not in BATCH_FAMILIES:
        return "this family keeps one clip per denoise"
    if getattr(fam, "modular_workflow", False):
        return "a modular workflow renders one clip per call"
    if getattr(state, "transformer_cache", None):
        # A step cache decides on the whole batch's residual, so one clip could change another's skips.
        return "a step cache is on"
    if getattr(state, "vram_split_cards", None):
        return "the load is split across cards"
    pipe = getattr(state, "pipe", None)
    # A compiled or graph-captured load batches eager (eager_render): never a compile or a capture on a
    # user's request. Measured on Wan 2.2 5B (A100-40, 960x544x33, 12 steps): one compiled clip replays
    # faster than a batch runs eager (7.7 vs 16.3 s per clip on the card, 12.2 vs 17.0 streamed), so
    # Auto keeps such loads serial; force batches them eager.
    eager = eager_batch_available()
    if getattr(pipe, "_unsloth_cuda_graphs", None) and (chosen == "auto" or not eager):
        return "CUDA graphs replay one clip faster than a batch runs eager"
    try:
        from .diffusion_cfg_batch import compiled_denoiser, per_call_scaled
    except Exception:  # noqa: BLE001 -- no torch here: nothing to batch
        return "torch is unavailable"
    for module in _denoisers(pipe):
        why = compiled_denoiser(module)
        if why is not None and (chosen == "auto" or not eager):
            return f"{why}: one compiled clip is faster than a batch run eager"
        try:
            scaled = per_call_scaled(module)
        except Exception:  # noqa: BLE001 -- a fake denoiser has no modules
            scaled = None
        if scaled is not None:
            # One activation scale over every row: a neighbour would change this clip's numbers.
            return f"activation scales come from the whole call ({scaled})"
    noisy = stochastic_sampler(getattr(pipe, "scheduler", None))
    if noisy is not None:
        # The pipeline steps it without a generator: the batch would draw every row's noise from one stream.
        return f"the sampler draws fresh noise every step ({noisy})"
    if chosen == "auto" and not streamed_load(state):
        # Measured: a denoiser kept on the card is compute bound, so a batch is no faster than serial.
        return "only a Low VRAM load is measured faster batched"
    return None


def stochastic_sampler(scheduler: Any) -> Optional[str]:
    """Why ``scheduler.step`` draws noise each step, else None. Only the initial noise is per row (one generator
    each); a sampler that renoises every step from a shared stream would shift a clip's draws with the batch."""
    if scheduler is None:
        return None
    name = type(scheduler).__name__
    if any(tag in name for tag in ("Ancestral", "SDE", "LCM", "DDPM", "Stochastic")):
        return name
    config = getattr(scheduler, "config", None)
    get = getattr(config, "get", None)
    if callable(get):
        if get("stochastic_sampling", False):
            return f"{name} stochastic_sampling"
        algorithm = str(get("algorithm_type", "") or "")
        if "sde" in algorithm.lower():
            return f"{name} {algorithm}"
    return None


def eager_batch_available() -> bool:
    """A batch on a compiled or graph-captured load runs eager (``eager_render``): never a compile or a capture
    on a user's request. Needs ``torch.compiler.set_stance`` (torch 2.6+)."""
    try:
        import torch
        return callable(getattr(getattr(torch, "compiler", None), "set_stance", None))
    except Exception:  # noqa: BLE001
        return False


def eager_render(pipe: Any):
    """While active, compiled blocks run their eager code and captured CUDA graphs are bypassed (kept, not
    replayed): the batch shape is new to both, and neither may compile or capture on a user's request."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        handles = tuple(getattr(pipe, "_unsloth_cuda_graphs", ()) or ())
        before = [getattr(handle, "bypassed", False) for handle in handles]
        if handles:
            from . import diffusion_cuda_graph
            diffusion_cuda_graph.set_bypass(handles, True)
        compiled = any(_compiled(module) for module in _denoisers(pipe))
        stance = contextlib.nullcontext()
        if compiled:
            import torch
            stance = torch.compiler.set_stance("force_eager")
        try:
            with stance:
                yield
        finally:
            for handle, was in zip(handles, before):
                try:
                    handle.set_bypass(was)
                except Exception:  # noqa: BLE001 -- optimisation only
                    pass

    return _ctx()


def _compiled(module: Any) -> bool:
    try:
        from .diffusion_cfg_batch import compiled_denoiser
        return compiled_denoiser(module) is not None
    except Exception:  # noqa: BLE001
        return False


def streamed_load(state: Any) -> bool:
    """Low VRAM: the denoiser streams its weights for every call. The one placement measured faster batched, on
    Wan 2.2 5B (A100-40, 960x544x33, 12 steps, eager, CFG 5): a batch of 4 ran 1.65x to 1.72x the clips per
    minute of serial, every batched clip byte-identical to its solo render. The default group offload (its groups
    mostly stay on the card) measured 1.00x on the A100-40 and 0.78x to 0.92x on a T4; other tiers are unmeasured."""
    return str(getattr(state, "memory_mode", "") or "") == "low_vram"


def request_decline_reason(
    *,
    first_frame: Any = None,
    last_frame: Any = None,
    references: Any = None,
    continuation: Any = None,
    draft_decode: Optional[bool] = None,
) -> Optional[str]:
    """Why this request renders alone, else None."""
    if first_frame is not None or last_frame is not None:
        return "image-conditioned clips render alone"
    if references:
        return "reference-conditioned clips render alone"
    if continuation is not None:
        return "continuations render alone"
    try:
        from . import video_tiny_vae
        if video_tiny_vae.draft_decode_wanted(draft_decode):
            return "a draft decode renders alone"
    except Exception:  # noqa: BLE001 -- no draft decoder module: nothing to refuse
        pass
    return None


def request_key(
    *,
    width: Any,
    height: Any,
    num_frames: Any,
    fps: Any,
    steps: Any,
    guidance: Any,
    guidance_2: Any,
    flow_shift: Any,
) -> tuple:
    """What every row of one denoise shares: the latent shape, the timestep grid and the guidance scale."""

    def _f(value: Any) -> Any:
        return None if value is None else float(value)

    def _i(value: Any) -> Any:
        return None if value is None else int(value)

    return (
        _i(width),
        _i(height),
        _i(num_frames),
        _i(fps),
        _i(steps),
        _f(guidance),
        _f(guidance_2),
        _f(flow_shift),
    )


class BatchDoesNotFit(RuntimeError):
    """The batch's denoise would not fit beside the load's weights; its clips render one at a time instead."""


@dataclass
class Rider:
    """One clip denoised in another clip's batch. ``outcome`` is its terminal progress record."""

    video_id: Optional[str]
    account_id: Optional[str]
    prompt: str
    negative_prompt: Optional[str]
    seed: Optional[int]
    cancel: threading.Event
    record_outcome: bool = True
    gen_kwargs: dict = field(default_factory = dict)
    outcome: Optional[dict] = None
    # Set once its clip is being saved: a Stop after that point is too late to discard it.
    closed: bool = False
    # The AccountContext that queued it: its clip and job outcome are persisted as this account, not the job's.
    account: Any = None
    # Its outcome is in its job store; a batch that ends records each rider not yet recorded.
    recorded: bool = False


@dataclass
class Forming:
    """The batch a started clip gathers while the queue hands over the compatible clips behind it."""

    key: tuple
    state: Any
    riders: list = field(default_factory = list)
    sealed: threading.Event = field(default_factory = threading.Event)
    limit: int = DEFAULT_MAX_ROWS
    # The job's own clip's Stop (begin_generate's event); the render watches every member's.
    leader_cancel: Optional[threading.Event] = None

    def room(self) -> bool:
        return not self.sealed.is_set() and 1 + len(self.riders) < self.limit


class AllCancelled:
    """A cancel event for the shared denoise: set only once every member asked to stop."""

    def __init__(
        self,
        leader: threading.Event,
        others: Callable[[], list],
        hard: Optional[Callable[[], bool]] = None,
    ) -> None:
        # The job's own clip's event: its Stop alone must not stop the riders.
        self.leader = leader
        self._others = others
        self._own = threading.Event()
        self._hard = hard

    def is_set(self) -> bool:
        if self._own.is_set():
            return True
        if self.leader.is_set() and all(event.is_set() for event in self._others()):
            return True
        return bool(self._hard is not None and self._hard())

    def set(self) -> None:
        self._own.set()

    def clear(self) -> None:
        self._own.clear()

    def wait(self, timeout: Optional[float] = None) -> bool:
        import time

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while not self.is_set():
            left = 0.05 if deadline is None else min(0.05, deadline - time.monotonic())
            if left <= 0:
                return self.is_set()
            self._own.wait(left)
        return True


def row_generators(seeds: list, device: Any) -> list:
    """One generator per row, seeded exactly as that clip's own render seeds its single generator."""
    import torch
    return [torch.Generator(device = device).manual_seed(int(seed)) for seed in seeds]


def per_row_encode(pipe: Any):
    """While active, a batched ``pipe.encode_prompt`` (a prompt list) encodes each clip's prompt and negative prompt
    alone and stacks the results: the text encoder sees one prompt per call, exactly as in that clip's own render,
    so its embeddings carry no batch-shape difference (UMT5 is where batched Wan CFG renders were seen to drift)."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        original = getattr(pipe, "encode_prompt", None)
        if not callable(original):
            yield
            return
        had_own = "encode_prompt" in getattr(pipe, "__dict__", {})

        def encode_prompt(
            prompt: Any = None,
            negative_prompt: Any = None,
            **kwargs: Any,
        ) -> Any:
            if not isinstance(prompt, (list, tuple)) or len(prompt) <= 1:
                return original(prompt = prompt, negative_prompt = negative_prompt, **kwargs)
            import torch

            negatives = (
                list(negative_prompt)
                if isinstance(negative_prompt, (list, tuple))
                else [negative_prompt] * len(prompt)
            )
            rows = [
                original(prompt = text, negative_prompt = negative, **kwargs)
                for text, negative in zip(prompt, negatives)
            ]
            positive = torch.cat([row[0] for row in rows], dim = 0)
            negative = None if rows[0][1] is None else torch.cat([row[1] for row in rows], dim = 0)
            return positive, negative

        encode_prompt.__wrapped__ = original
        try:
            pipe.encode_prompt = encode_prompt
        except Exception:  # noqa: BLE001 -- a pipe that refuses assignment encodes as one batch
            yield
            return
        try:
            yield
        finally:
            try:
                if had_own:
                    pipe.encode_prompt = original
                else:
                    del pipe.encode_prompt
            except Exception:  # noqa: BLE001 -- cleanup is best-effort
                pass

    return _ctx()


def per_row_decode(pipe: Any, rows: Callable[[], int]):
    """While active, a ``vae.decode`` of a batch decodes one clip at a time and joins the results, so the
    decode peak stays one clip's (decode is per request) and each clip's pixels come from a one-clip decode."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        vae = getattr(pipe, "vae", None)
        if vae is None or not callable(getattr(vae, "decode", None)) or rows() <= 1:
            yield
            return
        from . import decode_layers

        def make(inner: Any, _layer: Any) -> Any:
            def decode(z: Any, *args: Any, **kwargs: Any) -> Any:
                count = int(getattr(z, "shape", (1,))[0] or 1)
                if count <= 1:
                    return inner(z, *args, **kwargs)
                import torch

                outputs = [inner(z[i : i + 1], *args, **kwargs) for i in range(count)]
                first = outputs[0]
                if isinstance(first, tuple):
                    return (torch.cat([out[0] for out in outputs], dim = 0),) + tuple(first[1:])
                sample = getattr(first, "sample", None)
                if sample is not None:
                    first.sample = torch.cat([out.sample for out in outputs], dim = 0)
                    return first
                return torch.cat(outputs, dim = 0)

            return decode

        layer = decode_layers.wrap(vae, "decode", DECODE_LAYER, make)
        try:
            from .diffusion_vae_oom import ensure_outermost
            ensure_outermost(vae)
        except Exception:  # noqa: BLE001 -- no OOM scope on this decoder
            pass
        try:
            yield
        finally:
            decode_layers.unwrap(layer)
            try:
                from .diffusion_vae_oom import ensure_outermost
                ensure_outermost(vae)
            except Exception:  # noqa: BLE001
                pass

    return _ctx()
