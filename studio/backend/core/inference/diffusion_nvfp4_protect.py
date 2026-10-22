# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-step precision for the NVFP4 backend: which denoising steps run W4A16 instead of W4A4.

A protected step dequantises to a transient bf16 weight; the counter wraps ``pipe.scheduler.step``.
The set must be MEASURED per model (a copied set did worse than none), so it is off unless named.
"""

from __future__ import annotations

import contextlib
import math
import os
import weakref
from typing import Any, Optional

PROTECT_STEPS_ENV = "UNSLOTH_NVFP4_PROTECT_STEPS"

AUTO = "auto"
AUTO_HEAD_FRACTION = 0.08

ALL = "all"

_OFF_TOKENS = ("", "off", "none", "0-none", "false", "no")


def protect_steps_env() -> str:
    raw = os.environ.get(PROTECT_STEPS_ENV, "").strip().lower()
    return "" if raw in _OFF_TOKENS else raw


def parse_protect_steps(spec: Any, total_steps: int) -> tuple:
    """``spec`` resolved against ``total_steps``, sorted; out-of-range drops, a non-integer token raises."""
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
    """Current step and whether it is protected; ``protected`` is a plain bool (a property recompiles per STEP)."""

    def __init__(self, spec: Any = None) -> None:
        self.spec: str = ""
        self.armed: bool = False
        self.total: int = 0
        self.steps: tuple = ()
        self.index: int = 0
        self.protected: bool = False
        self.protected_steps_seen: int = 0
        self.generations: int = 0
        self._layers: "weakref.WeakSet" = weakref.WeakSet()
        self.configure(protect_steps_env() if spec is None else spec)

    def register_layer(self, layer: Any) -> None:
        try:
            self._layers.add(layer)
        except TypeError:  # an unweakrefable layer simply is not counted
            pass

    def capable_layers(self) -> int:
        return len(self._layers)

    def configure(self, spec: Any) -> "NVFP4StepController":
        """Set the schedule before the first forward: ``armed`` is a compile guard."""
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
        self.index += 1
        self.protected = self.index in self.steps
        if self.protected:
            self.protected_steps_seen += 1
        return self.index

    def reset(self) -> "NVFP4StepController":
        self.total = 0
        self.steps = ()
        self.index = 0
        self.protected = False
        return self


_CONTROLLER = NVFP4StepController()


def protect_controller() -> NVFP4StepController:
    return _CONTROLLER


def reset_protect_controller(spec: Any = None) -> NVFP4StepController:
    return _CONTROLLER.configure(protect_steps_env() if spec is None else spec)


@contextlib.contextmanager
def protect_generation(
    pipe: Any,
    steps: int,
    *,
    controller: Optional[NVFP4StepController] = None,
    logger: Any = None,
):
    """Drive ``controller`` across one generation, then restore; no scheduler protects NOTHING."""
    ctls = [controller] if controller is not None else pipeline_controllers(pipe)
    ctl = ctls[0]
    if not ctl.armed:
        yield ctl
        return
    if not sum(c.capable_layers() for c in ctls):
        if logger is not None:
            logger.warning(
                "[nvfp4] protect schedule %r requested but no protect-capable NVFP4 layer is "
                "loaded (the torchao backend has no W4A16 branch); the lever stays off for this "
                "generation",
                ctl.spec,
            )
        for c in ctls:
            c.reset()
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
        for c in ctls:
            c.reset()
        yield ctl
        return

    for i, c in enumerate(ctls):
        c.begin(steps, logger = logger if i == 0 else None)

    def _step(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        for c in ctls:
            c.advance()
        return out

    # Delete, not reassign: an instance attribute would shadow the class method forever.
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
        for c in ctls:
            c.reset()


@contextlib.contextmanager
def suspend_protect(modules: Any):
    """The prewarm MUST suspend the lever, or it tunes the bf16 branch and marks the FP4 shape tuned."""
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


def protect_graph_key(
    protected: Optional[bool] = None,
    module: Any = None,
    *,
    controller: Optional[NVFP4StepController] = None,
) -> tuple:
    """CUDA-graph key suffix for the branch in flight (a graph records ONE branch), or ``()`` when off."""
    ctl = controller if controller is not None else module_controller(module)
    if not ctl.armed:
        return ()
    live = ctl.protected if protected is None else bool(protected)
    return (("nvfp4_protect", bool(live)),)


def protect_layers(module: Any) -> list:
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
    layers = protect_layers(module)
    for _, layer in layers:
        old = layer.protect
        if old is not controller:
            old._layers.discard(layer)
        layer.protect = controller
        controller.register_layer(layer)
    return len(layers)


def attach_own_controller(module: Any) -> Optional[NVFP4StepController]:
    """Per-model controller, so concurrent image and video renders never move each other's steps."""
    ctl = NVFP4StepController(protect_controller().spec)
    return ctl if attach_controller(module, ctl) else None


def module_controller(module: Any) -> NVFP4StepController:
    if module is not None:
        for _, layer in protect_layers(module):
            return layer.protect
    return protect_controller()


_DENOISER_ATTRS = ("transformer", "transformer_2", "unet")


def pipeline_controllers(pipe: Any) -> list:
    found: list = []
    for attr in _DENOISER_ATTRS:
        module = getattr(pipe, attr, None)
        for _, layer in protect_layers(module) if module is not None else ():
            if all(layer.protect is not c for c in found):
                found.append(layer.protect)
    return found or [protect_controller()]
