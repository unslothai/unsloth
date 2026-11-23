# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Few-step DMD sampling for distribution-matching distilled Wan students (FastVideo FastWan, dense attention).

A DMD student is trained at a fixed list of timesteps. Each step it predicts the clean latent from its flow
output, ``x0 = x_t - sigma_t * v``; between steps the prediction is re-noised to the next timestep with fresh
seeded noise, ``x_next = (1 - sigma_next) * x0 + sigma_next * eps``; the last step returns ``x0``. There is no
classifier-free guidance. The scheduler below implements exactly that behind the diffusers scheduler interface
(``set_timesteps`` / ``timesteps`` / ``sigmas`` / ``step``), so the Wan pipeline's own loop, the step callback,
compile, CUDA graphs, quantisation and offload paths run unchanged; only ``pipe.scheduler`` is swapped for the
render."""

from __future__ import annotations

import contextlib
import re
from typing import Any, Iterator, Optional

# The sigma table the students were distilled on: 1000 training timesteps under flow shift 8, each request
# timestep snapped to its nearest entry (independent of the inference flow_shift in the repo's scheduler config).
DMD_TRAINING_SHIFT = 8.0
DMD_TRAIN_TIMESTEPS = 1000

# (last-path-segment regex on the normalised name, trained timesteps); every other FastWan release stays refused
# by video_families.
_DMD_RECIPES: tuple[tuple[str, tuple[int, ...]], ...] = (
    (r"(?<![a-z0-9])fastwan-?2-2-ti2v-5b-fullattn(?![a-z0-9])", (1000, 757, 522)),
    # Distilled with sparse attention; sampled dense it matches the 30-step base on CLIP-T / PickScore (L4, 8 prompts).
    (r"(?<![a-z0-9])fastwan-?2-1-t2v-1-3b(?![a-z0-9])", (1000, 757, 522)),
)
DMD_STEPS_DEFAULT = 3

# Engagement counter (renders that ran under the DMD scheduler); read by tests and the GPU bench.
_ENGAGED = {"renders": 0, "steps": 0}


def _normalise(identifier: str) -> str:
    return re.sub(r"[\s._]+", "-", str(identifier).lower())


def dmd_timesteps(*identifiers: Optional[str]) -> Optional[tuple[int, ...]]:
    """The trained timesteps when the first identifier naming a model is a known DMD student, else None."""
    for identifier in identifiers:
        if not identifier:
            continue
        from .family_name_match import model_identity

        # The same identity the family / defaults matchers read: an HF cache snapshot path names its repo.
        name = re.split(r"[/\\]", _normalise(model_identity(identifier)).rstrip("/\\"))[-1]
        for pattern, timesteps in _DMD_RECIPES:
            if re.search(pattern, name):
                return timesteps
    return None


def training_sigma_table():
    """``(sigmas, timesteps)`` float64 of the DMD training table, highest noise first."""
    import torch

    s = torch.linspace(1.0, float(DMD_TRAIN_TIMESTEPS), DMD_TRAIN_TIMESTEPS, dtype = torch.float64)
    s = s.flip(0) / DMD_TRAIN_TIMESTEPS
    shift = DMD_TRAINING_SHIFT
    sigmas = shift * s / (1.0 + (shift - 1.0) * s)
    return sigmas, sigmas * DMD_TRAIN_TIMESTEPS


def dmd_sigmas(timesteps: tuple[int, ...]) -> list[float]:
    """Sigma per trained timestep: the training-table entry whose timestep is nearest."""
    import torch

    sigmas, table = training_sigma_table()
    t = torch.tensor([float(x) for x in timesteps], dtype = torch.float64)
    idx = (table.unsqueeze(0) - t.unsqueeze(1)).abs().argmin(dim = 1)
    return [float(v) for v in sigmas[idx]]


def renoise_seed(seed: int) -> int:
    """The re-noise stream's seed: derived from the request seed (deterministic per request) but never equal to it.
    The pipeline draws the initial latent from a generator seeded with the request seed (a CPU one on CPU / MPS), so
    seeding the re-noise stream with it too would make the first re-noise draw repeat the initial noise exactly."""
    import hashlib

    digest = hashlib.sha256(f"unsloth-dmd-renoise:{int(seed)}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _scheduler_base():
    from diffusers import FlowMatchEulerDiscreteScheduler
    return FlowMatchEulerDiscreteScheduler


def make_dmd_scheduler(
    timesteps: tuple[int, ...],
    seed: int,
    *,
    device: Any = None,
) -> Any:
    """A diffusers-compatible scheduler running the DMD update over ``timesteps``, noise from a CPU generator
    seeded with :func:`renoise_seed` of ``seed`` (identical across CUDA, ROCm, MPS and CPU, independent of the
    initial latent noise)."""
    import torch

    base = _scheduler_base()

    class DmdFlowScheduler(base):  # type: ignore[misc, valid-type]
        _unsloth_dmd = True

        def __init__(self) -> None:
            super().__init__(num_train_timesteps = DMD_TRAIN_TIMESTEPS, shift = 1.0)
            self._dmd_timesteps = tuple(int(x) for x in timesteps)
            self._dmd_sigmas = dmd_sigmas(self._dmd_timesteps)
            self._dmd_seed = renoise_seed(seed)
            self._dmd_generator = torch.Generator(device = "cpu")
            self.set_timesteps(len(self._dmd_timesteps), device = device)

        def set_shift(self, shift: float) -> None:
            # The trained timesteps fix the noise levels; a request flow shift has nothing to move.
            return None

        def set_timesteps(
            self,
            num_inference_steps: Any = None,
            device: Any = None,
            **_kw: Any,
        ) -> None:
            dev = device if device is not None else "cpu"
            self.timesteps = torch.tensor(
                [float(t) for t in self._dmd_timesteps], dtype = torch.float32, device = dev
            )
            self.sigmas = torch.tensor([*self._dmd_sigmas, 0.0], dtype = torch.float32, device = dev)
            self.num_inference_steps = len(self._dmd_timesteps)
            # Reseeded per render, so an OOM retry of the same request draws the same noise.
            self._dmd_generator.manual_seed(self._dmd_seed)
            self._dmd_index = 0
            self._step_index = None
            self._begin_index = None

        def step(
            self,
            model_output: Any,
            timestep: Any,
            sample: Any,
            *args: Any,
            return_dict: bool = True,
            **kwargs: Any,
        ) -> Any:
            i = self._dmd_index
            sigma = self._dmd_sigmas[i]
            x0 = sample.to(torch.float32) - sigma * model_output.to(torch.float32)
            if i + 1 < len(self._dmd_sigmas):
                nxt = self._dmd_sigmas[i + 1]
                noise = torch.randn(
                    tuple(sample.shape), generator = self._dmd_generator, dtype = torch.float32
                ).to(sample.device)
                out = (1.0 - nxt) * x0 + nxt * noise
            else:
                out = x0
            self._dmd_index = i + 1
            self._step_index = i + 1
            _ENGAGED["steps"] += 1
            out = out.to(sample.dtype)
            if not return_dict:
                return (out,)
            from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
                FlowMatchEulerDiscreteSchedulerOutput,
            )

            return FlowMatchEulerDiscreteSchedulerOutput(prev_sample = out)

    return DmdFlowScheduler()


@contextlib.contextmanager
def dmd_sampling(
    pipe: Any,
    timesteps: tuple[int, ...],
    seed: int,
    *,
    logger: Any = None,
) -> Iterator[Any]:
    """Swap ``pipe.scheduler`` for the DMD scheduler for one render; the original is always restored."""
    original = getattr(pipe, "scheduler", None)
    device = getattr(pipe, "_execution_device", None)
    scheduler = make_dmd_scheduler(timesteps, seed, device = device)
    pipe.scheduler = scheduler
    _ENGAGED["renders"] += 1
    if logger is not None:
        logger.info(
            "video.dmd: %d-step DMD sampling at timesteps %s (sigmas %s), CFG off",
            len(timesteps),
            list(timesteps),
            [round(s, 4) for s in scheduler._dmd_sigmas],
        )
    try:
        yield scheduler
    finally:
        pipe.scheduler = original


def engaged_counts(reset: bool = False) -> dict:
    out = dict(_ENGAGED)
    if reset:
        _ENGAGED["renders"] = 0
        _ENGAGED["steps"] = 0
    return out
