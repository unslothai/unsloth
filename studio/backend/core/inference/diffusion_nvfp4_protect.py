# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-step precision for the NVFP4 backend: which denoising steps run W4A16 instead of W4A4.

A DiT runs the same weights 40-100 times in a feedback loop, and the damage the 4-bit activation
path does is nowhere near uniform across those calls. Step 0 carries by far the most of it. The
lever this module drives is therefore to run the SAME 4-bit bytes at a higher activation precision
on a named handful of steps: at a protected step ``NVFP4FlashInferLinear`` dequantises ``wq`` /
``w_sf`` / ``w_scale`` to a transient bf16 weight and runs a plain ``F.linear``; everywhere else it
runs the FlashInfer NVFP4 GEMM exactly as before.

**Zero extra resident memory.** The bf16 weight is built inside the forward and dropped when it
returns; there is never a second resident operand and the checkpoint is byte-identical to the one
the plain flashinfer arm loads. That is the whole point of switching on the ACTIVATION side.

Three things live here:

1. **The schedule.** ``UNSLOTH_NVFP4_PROTECT_STEPS`` is a comma list of step indices, negative
   indices counting from the end (``0,1,2,3,-1``), or ``auto`` for "the first 8 percent of the
   steps plus the last one", which is ``0,1,2,3,49`` on the 50-step video schedule. Empty (the
   default) is OFF and the layer keeps the forward it has today, guard for guard.
2. **The controller.** One small object per process holding a plain Python ``bool``. The layer
   reads ``ctl.armed and ctl.protected``, which under Dynamo is two constant guards and therefore
   at most TWO compiled variants of a block for a whole render, not one per step. ``armed`` is
   fixed for the life of a load (it comes from the environment), so a load that never asked for
   the lever traces exactly one variant: the short circuit means ``protected`` is never read and
   never guarded.
3. **The step counter.** ``protect_generation`` wraps ``pipe.scheduler.step`` for the duration of
   one generation. Every diffusers denoise loop calls it exactly once per step, AFTER the
   transformer forward for that step, which is what makes a counter incremented there the index of
   the forward that is about to run. It is the same hook ``video._scheduler_step_progress`` and the
   gate harness's step timer already use, and it works for the pipelines that expose
   ``callback_on_step_end`` and the ones that do not, which is why the lever is wired there rather
   than to a callback only half the families accept.

The protected set must be MEASURED per model. Copying one model's set to another has been observed
to make held-out prompts worse than protecting nothing, so there is no per-family default here and
the lever is off unless an operator names the steps.
"""

from __future__ import annotations

import contextlib
import math
import os
from typing import Any, Optional

# The schedule. Empty / unset / "off" is OFF, which is the default everywhere.
PROTECT_STEPS_ENV = "UNSLOTH_NVFP4_PROTECT_STEPS"

# ``auto``: the first ``AUTO_HEAD_FRACTION`` of the schedule, plus the last step. 0.08 of 50 is 4,
# i.e. ``0,1,2,3,-1``, which is the set the video campaign measured on Wan2.2-TI2V-5B; the fraction
# is what carries it to a schedule with a different step count.
AUTO = "auto"
AUTO_HEAD_FRACTION = 0.08

_OFF_TOKENS = ("", "off", "none", "0-none", "false", "no")


def protect_steps_env() -> str:
    """The requested schedule, verbatim and lowercased. ``""`` means the lever is off."""
    raw = os.environ.get(PROTECT_STEPS_ENV, "").strip().lower()
    return "" if raw in _OFF_TOKENS else raw


def parse_protect_steps(spec: Any, total_steps: int) -> tuple:
    """``spec`` resolved against a schedule of ``total_steps`` steps, as a sorted tuple.

    Negative indices count from the end, so one environment value serves every step count that
    wants "the first few and the last". An index the schedule does not REACH is dropped rather than
    raising: the same value has to survive a 4-step smoke render and a 50-step production one. A
    token that is not an integer DOES raise, because that is a typo and silently protecting nothing
    is how a lever gets reported as measured when it never ran.
    """
    total = int(total_steps)
    if total <= 0:
        return ()
    raw = "" if spec is None else str(spec).strip().lower()
    if raw in _OFF_TOKENS:
        return ()
    if raw == AUTO:
        head = int(math.ceil(AUTO_HEAD_FRACTION * total))
        wanted: list = list(range(min(head, total))) + [total - 1]
    else:
        wanted = []
        for token in raw.replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                wanted.append(int(token))
            except ValueError:
                raise ValueError(
                    f"{PROTECT_STEPS_ENV}={spec!r}: {token!r} is not a step index. Use a comma "
                    f"list of integers (negative counts from the end, e.g. '0,1,2,3,-1'), "
                    f"'{AUTO}', or leave it empty for off"
                ) from None
    resolved = set()
    for index in wanted:
        if index < 0:
            index += total
        if 0 <= index < total:
            resolved.add(index)
    return tuple(sorted(resolved))


class NVFP4StepController:
    """Which denoising step is running, and whether it is protected.

    ``protected`` is a plain ``bool`` attribute rather than a property on purpose: Dynamo turns a
    constant attribute read into one guard and compiles one variant per value, so a render traces
    two variants of a protected block and no more. A property computing ``index in steps`` would
    guard the index instead, which is one variant per STEP.
    """

    def __init__(self, spec: Any = None) -> None:
        self.spec: str = ""
        self.armed: bool = False
        self.total: int = 0
        self.steps: tuple = ()
        self.index: int = 0
        self.protected: bool = False
        # Diagnostics, so a run can prove the lever fired without reading the renders.
        self.protected_steps_seen: int = 0
        self.generations: int = 0
        self.configure(protect_steps_env() if spec is None else spec)

    # ── configuration ────────────────────────────────────────────────────────────────────────
    def configure(self, spec: Any) -> "NVFP4StepController":
        """Set the schedule. ``armed`` is fixed here and read as a compile guard by the layer, so
        it must not move once a load has traced: configure before the first forward."""
        raw = "" if spec is None else str(spec).strip().lower()
        self.spec = "" if raw in _OFF_TOKENS else raw
        self.armed = bool(self.spec)
        self.reset()
        return self

    def describe(self) -> dict:
        return {
            "env": PROTECT_STEPS_ENV,
            "spec": self.spec,
            "armed": self.armed,
            "total_steps": self.total,
            "protected_steps": list(self.steps),
            "protected_steps_seen": self.protected_steps_seen,
            "generations": self.generations,
        }

    # ── the per-generation state machine ─────────────────────────────────────────────────────
    def begin(self, total_steps: int, *, logger: Any = None) -> tuple:
        """Start a generation of ``total_steps`` steps. Returns the resolved protected set."""
        self.reset()
        if not self.armed:
            return ()
        try:
            self.steps = parse_protect_steps(self.spec, total_steps)
        except ValueError as exc:  # a typo must not cost a multi-minute render
            self.steps = ()
            if logger is not None:
                logger.warning("[nvfp4] protect schedule ignored: %s", exc)
        self.total = int(total_steps)
        self.index = 0
        self.protected = 0 in self.steps
        self.protected_steps_seen = 1 if self.protected else 0
        self.generations += 1
        if logger is not None and self.steps:
            logger.info(
                "[nvfp4] W4A16 protect: steps %s of %d run the dequantised bf16 GEMM",
                ",".join(str(s) for s in self.steps),
                self.total,
            )
        return self.steps

    def advance(self) -> int:
        """One denoising step finished. Returns the index of the step about to run."""
        self.index += 1
        self.protected = self.index in self.steps
        if self.protected:
            self.protected_steps_seen += 1
        return self.index

    def reset(self) -> "NVFP4StepController":
        """Back to "no generation in flight", which protects nothing."""
        self.total = 0
        self.steps = ()
        self.index = 0
        self.protected = False
        return self


# One controller per process. Studio serves one generation at a time behind its load lock, and the
# layers take this object at construction, so a per-model controller buys nothing today; injecting
# one (assign ``layer.protect``) is still how the tests drive a layer without touching the process.
_CONTROLLER = NVFP4StepController()


def protect_controller() -> NVFP4StepController:
    """The process-wide controller. Every converted layer reads this one object."""
    return _CONTROLLER


def reset_protect_controller(spec: Any = None) -> NVFP4StepController:
    """Re-read the environment (or take ``spec``) and clear any generation state. For tests, and
    for a load that changed the schedule under a process that had already read it."""
    return _CONTROLLER.configure(protect_steps_env() if spec is None else spec)


@contextlib.contextmanager
def protect_generation(
    pipe: Any,
    steps: int,
    *,
    controller: Optional[NVFP4StepController] = None,
    logger: Any = None,
):
    """Drive ``controller`` across one generation of ``steps`` steps, then restore everything.

    A no-op context when the lever is off: nothing is wrapped, nothing is counted, and the
    scheduler is left exactly as it was found. When it is on, ``pipe.scheduler.step`` is wrapped
    for the duration -- it is called once per denoise step, after that step's transformer forward,
    so incrementing there leaves the counter holding the index of the NEXT forward. A pipeline with
    no scheduler to count (a modular workflow that hides it) protects NOTHING rather than
    everything, and says so.
    """
    ctl = controller if controller is not None else protect_controller()
    if not ctl.armed:
        yield ctl
        return
    scheduler = getattr(pipe, "scheduler", None)
    original = getattr(scheduler, "step", None)
    if scheduler is None or not callable(original):
        if logger is not None:
            logger.warning(
                "[nvfp4] protect schedule requested but this pipeline exposes no scheduler.step "
                "to count denoising steps; the lever stays off for this generation"
            )
        ctl.reset()
        yield ctl
        return

    ctl.begin(steps, logger = logger)

    def _step(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        ctl.advance()
        return out

    # Whether ``step`` was an instance attribute BEFORE this wrap. If it was not (the ordinary
    # case: a bound class method), restoring by assignment would leave an instance attribute
    # shadowing the class for the life of the scheduler, so unwind by deleting instead. Nesting
    # under another wrapper -- the video backend's progress hook, the gate harness's step timer --
    # is the case where it WAS one, and that callable is put back exactly.
    had_own = "step" in getattr(scheduler, "__dict__", {})
    scheduler.step = _step
    try:
        yield ctl
    finally:
        if had_own:
            scheduler.step = original
        else:
            try:
                del scheduler.step
            except (AttributeError, TypeError):  # noqa: PERF203 - a slotted or proxied scheduler
                scheduler.step = original
        ctl.reset()


def protect_graph_key(protected: Optional[bool] = None) -> tuple:
    """The CUDA-graph cache-key suffix for the branch in flight, or ``()`` when the lever is off.

    A captured graph records ONE branch. Without this the graph taken at an unprotected step would
    replay at a protected one and the lever would be silently inert under capture, which is worse
    than refusing to capture: the numbers would look like the lever ran. With it each branch gets
    its own graph, at the cost of doubling the graph count for a load that arms the lever.
    """
    ctl = protect_controller()
    if not ctl.armed:
        return ()
    live = ctl.protected if protected is None else bool(protected)
    return (("nvfp4_protect", bool(live)),)


def protect_layers(module: Any) -> list:
    """``(fqn, layer)`` for every NVFP4 layer under ``module`` that can take the branch."""
    from .diffusion_nvfp4_linear import is_nvfp4_flashinfer_linear

    found: list = []
    named = getattr(module, "named_modules", None)
    if not callable(named):
        return found
    for name, sub in named():
        if is_nvfp4_flashinfer_linear(sub) and getattr(sub, "protect", None) is not None:
            found.append((name, sub))
    return found


def attach_controller(module: Any, controller: NVFP4StepController) -> int:
    """Point every NVFP4 layer under ``module`` at ``controller``. Returns how many. Tests use it;
    the load path does not need it, since the layers take the process controller at construction."""
    layers = protect_layers(module)
    for _, layer in layers:
        layer.protect = controller
    return len(layers)
