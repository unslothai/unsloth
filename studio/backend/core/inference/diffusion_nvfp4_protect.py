# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-step precision for the NVFP4 backend: which denoising steps run W4A16 instead of W4A4.

At a protected step the layer dequantises its own bytes to a transient bf16 weight, so there is
never a second resident operand. The step counter wraps ``pipe.scheduler.step``, which every
diffusers denoise loop calls once per step, after that step's forward. The protected set must be
MEASURED per model: copying one model's set to another has been observed to make held-out prompts
worse than protecting nothing, so the lever is off unless an operator names the steps.
"""

from __future__ import annotations

import contextlib
import math
import os
import weakref
from typing import Any, Optional

PROTECT_STEPS_ENV = "UNSLOTH_NVFP4_PROTECT_STEPS"

# ``auto``: the first ``AUTO_HEAD_FRACTION`` of the schedule, plus the last step.
AUTO = "auto"
AUTO_HEAD_FRACTION = 0.08

ALL = "all"

_OFF_TOKENS = ("", "off", "none", "0-none", "false", "no")


def protect_steps_env() -> str:
    """The requested schedule, verbatim and lowercased. ``""`` means the lever is off."""
    raw = os.environ.get(PROTECT_STEPS_ENV, "").strip().lower()
    return "" if raw in _OFF_TOKENS else raw


def parse_protect_steps(spec: Any, total_steps: int) -> tuple:
    """``spec`` resolved against a schedule of ``total_steps`` steps, as a sorted tuple. An
    out-of-range index is dropped, so one value serves any step count; a non-integer token DOES
    raise, since a typo that silently protects nothing reads as a measured lever that never ran."""
    total = int(total_steps)
    if total <= 0:
        return ()
    raw = "" if spec is None else str(spec).strip().lower()
    if raw in _OFF_TOKENS:
        return ()
    if raw == ALL:
        return tuple(range(total))
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
    """Which denoising step is running, and whether it is protected. ``protected`` is a plain
    ``bool``, never a property: a property would make Dynamo compile one variant per STEP."""

    def __init__(self, spec: Any = None) -> None:
        self.spec: str = ""
        self.armed: bool = False
        self.total: int = 0
        self.steps: tuple = ()
        self.index: int = 0
        self.protected: bool = False
        self.protected_steps_seen: int = 0
        self.generations: int = 0
        # Weak, so an unloaded model's layers stop counting on their own.
        self._layers: "weakref.WeakSet" = weakref.WeakSet()
        self.configure(protect_steps_env() if spec is None else spec)

    def register_layer(self, layer: Any) -> None:
        """Record that ``layer`` can take the W4A16 branch."""
        try:
            self._layers.add(layer)
        except TypeError:  # an unweakrefable layer simply is not counted
            pass

    def capable_layers(self) -> int:
        """How many live layers can take the protected branch."""
        return len(self._layers)

    def configure(self, spec: Any) -> "NVFP4StepController":
        """Set the schedule. ``armed`` is read as a compile guard, so it must not move once a load
        has traced: configure before the first forward."""
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
            "capable_layers": self.capable_layers(),
        }

    def begin(
        self,
        total_steps: int,
        *,
        logger: Any = None,
    ) -> tuple:
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


# One controller per process: Studio serves one generation at a time behind its load lock.
_CONTROLLER = NVFP4StepController()


def protect_controller() -> NVFP4StepController:
    """The process-wide controller. Every converted layer reads this one object."""
    return _CONTROLLER


def reset_protect_controller(spec: Any = None) -> NVFP4StepController:
    """Re-read the environment (or take ``spec``) and clear any generation state."""
    return _CONTROLLER.configure(protect_steps_env() if spec is None else spec)


@contextlib.contextmanager
def protect_generation(
    pipe: Any,
    steps: int,
    *,
    controller: Optional[NVFP4StepController] = None,
    logger: Any = None,
):
    """Drive ``controller`` across one generation of ``steps`` steps, then restore everything. A
    no-op when the lever is off, and a pipeline with no scheduler to count protects NOTHING."""
    ctl = controller if controller is not None else protect_controller()
    if not ctl.armed:
        yield ctl
        return
    if not ctl.capable_layers():
        # Only NVFP4FlashInferLinear consults the controller: a torchao load runs W4A4 at every step
        # whatever the schedule says, and must not be reported as protected.
        if logger is not None:
            logger.warning(
                "[nvfp4] protect schedule %r requested but no protect-capable NVFP4 layer is "
                "loaded (the torchao backend has no W4A16 branch); the lever stays off for this "
                "generation",
                ctl.spec,
            )
        ctl.reset()
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

    # Restoring a bound class method by assignment would leave an instance attribute shadowing the
    # class forever, so unwind by deleting unless there really was one before this wrap.
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


@contextlib.contextmanager
def suspend_protect(modules: Any):
    """Force every controller reachable from ``modules`` to report itself unarmed, then restore.
    The prewarm MUST suspend the lever: otherwise its forwards take the bf16 branch, tune nothing,
    mark the shape tuned anyway, and the next capture records an untuned tactic."""
    seen: dict = {}
    for module in modules or ():
        ctl = getattr(module, "protect", None)
        if ctl is not None and id(ctl) not in seen:
            seen[id(ctl)] = (ctl, bool(getattr(ctl, "armed", False)))
    for ctl, _ in seen.values():
        ctl.armed = False
    try:
        yield
    finally:
        for ctl, armed in seen.values():
            ctl.armed = armed


def protect_graph_key(protected: Optional[bool] = None) -> tuple:
    """The CUDA-graph cache-key suffix for the branch in flight, or ``()`` when the lever is off.
    A captured graph records ONE branch, so without this the lever is silently inert under
    capture while the numbers look like it ran."""
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
    """Point every NVFP4 layer under ``module`` at ``controller``. Returns how many."""
    layers = protect_layers(module)
    for _, layer in layers:
        layer.protect = controller
        controller.register_layer(layer)
    return len(layers)
