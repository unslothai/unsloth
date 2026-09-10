# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GPTQ correction and activation-scale baking for the NVFP4 layers of a prequant build.

Two build-time passes over a DENSE pipeline: GPTQ corrects each weight onto the same NVFP4 grid
using the layer's own input second moment, and one ``a_gsf = 6 * 448 / act_amax`` per layer is
measured here rather than at run time. This module never falls back to another quantiser; it raises
rather than shipping a weight at plain RTN with nothing saying so.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

FP4_MAX = 6.0
FP8_MAX = 448.0

# Damping ladder for the Cholesky, smallest first: the Hessian is positive SEMI-definite at best,
# so the answer to a failed factorisation is more damping, not another quantiser.
DAMP_LADDER: tuple = (0.01, 0.05, 0.1, 0.5, 1.0)


class GPTQFailure(RuntimeError):
    """The correction could not be solved for a layer. Raised, never swallowed."""


def gptq_quantize_to_nvfp4(
    weight,
    hessian,
    block: int = 16,
    damp: float = 0.01,
):
    """GPTQ-correct a weight ONTO the NVFP4 grid. The output must land exactly on the format's
    grid, or the quantiser that packs the checkpoint silently undoes the correction."""
    import torch

    n, k = weight.shape
    dev = weight.device
    w = weight.float().clone()
    h = hessian.float().clone()
    idx = torch.arange(k, device = dev)
    dead = torch.diag(h) == 0
    h[idx[dead], idx[dead]] = 1.0
    w[:, dead] = 0.0
    h[idx, idx] += damp * torch.diag(h).mean()
    h = torch.linalg.cholesky(h)
    h = torch.cholesky_inverse(h)
    h = torch.linalg.cholesky(h, upper = True)

    lev = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device = dev)
    gscale = weight.abs().amax().float().clamp(min = 1e-8) / (FP4_MAX * FP8_MAX)
    for start in range(0, k, block):
        stop = min(start + block, k)
        tile = w[:, start:stop].clone()
        err = torch.zeros_like(tile)
        hd = torch.diag(h)[start:stop]
        # The format stores the block scale in e4m3, so round it here to stay representable.
        bs = (tile.abs().amax(-1, keepdim = True) / FP4_MAX / gscale).clamp(min = 1e-12)
        bs = bs.to(torch.float8_e4m3fn).float().clamp(min = 1e-12)
        step = (bs * gscale)[:, 0]
        for j in range(stop - start):
            col = tile[:, j]
            u = (col / step).clamp(-FP4_MAX, FP4_MAX)
            dq = lev[(u.abs().unsqueeze(-1) - lev).abs().argmin(-1)] * u.sign() * step
            e = (col - dq) / hd[j]
            tile[:, j] = dq
            if j + 1 < stop - start:
                tile[:, j + 1 :] -= e.unsqueeze(1) * h[start + j, start + j + 1 : stop].unsqueeze(0)
            err[:, j] = e
        w[:, start:stop] = tile
        if stop < k:
            w[:, stop:] -= err @ h[start:stop, stop:]
    return w.to(weight.dtype)


def rtn_quantize_to_nvfp4(weight, block: int = 16):
    """Round-to-nearest onto the same NVFP4 grid, as the comparison baseline for GPTQ."""
    import torch

    n, k = weight.shape
    dev = weight.device
    lev = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device = dev)
    gscale = weight.abs().amax().float().clamp(min = 1e-8) / (FP4_MAX * FP8_MAX)
    v = weight.float().reshape(n, k // block, block)
    bs = (v.abs().amax(-1, keepdim = True) / FP4_MAX / gscale).clamp(min = 1e-12)
    bs = bs.to(torch.float8_e4m3fn).float().clamp(min = 1e-12)
    step = bs * gscale
    u = (v / step).clamp(-FP4_MAX, FP4_MAX)
    q = lev[(u.abs().unsqueeze(-1) - lev).abs().argmin(-1)] * u.sign()
    return (q * step).reshape(n, k).to(weight.dtype)


def gptq_correct(
    weight,
    hessian,
    *,
    block: int = 16,
    damps: Sequence = DAMP_LADDER,
) -> tuple:
    """``(corrected_weight, damp)``, escalating the damping until the Cholesky factors: the last
    rung raises rather than shipping this weight at RTN."""
    last: Optional[Exception] = None
    for damp in damps:
        try:
            return gptq_quantize_to_nvfp4(weight, hessian, block = block, damp = float(damp)), float(
                damp
            )
        except Exception as exc:  # noqa: BLE001 - a non-factorable Hessian is the expected failure
            last = exc
    raise GPTQFailure(
        f"the GPTQ Cholesky did not factor at any damping in {tuple(float(d) for d in damps)} "
        f"({type(last).__name__}: {last}). The Hessian is degenerate: calibrate on more prompts or "
        "more steps. This layer is NOT silently left at round-to-nearest."
    )


class HessianAccumulator:
    """Per-layer ``sum(x^T x)`` in fp32 over the sampled steps of a calibration run. The budget is
    checked UP FRONT, and gating on ``active`` catches both CFG halves."""

    #: fp32 bytes the accumulator may hold across all layers before it refuses to attach.
    DEFAULT_BUDGET_BYTES = 24 * 1024**3

    def __init__(
        self,
        modules: Mapping,
        *,
        budget_bytes: int = DEFAULT_BUDGET_BYTES,
        device: Any = None,
    ) -> None:
        self.modules = dict(modules)
        self.budget_bytes = int(budget_bytes)
        self.device = device
        self.active = False
        self.hessians: dict = {}
        self.samples: dict = {}
        self.steps_seen = 0
        self._handles: list = []

    def planned_bytes(self) -> int:
        """What ``attach`` would allocate, before it allocates it."""
        return sum(4 * int(mod.in_features) ** 2 for mod in self.modules.values())

    def budget_refusal(self) -> Optional[str]:
        """Why this set cannot be accumulated within the budget, or None."""
        want = self.planned_bytes()
        if want <= self.budget_bytes:
            return None
        widest = sorted(
            ((int(mod.in_features), fqn) for fqn, mod in self.modules.items()), reverse = True
        )[:3]
        return (
            f"{len(self.modules)} layers need {want / 1e9:.1f} GB of fp32 Hessians, over the "
            f"{self.budget_bytes / 1e9:.1f} GB budget (widest: "
            + ", ".join(f"{fqn} K={k}" for k, fqn in widest)
            + "). Raise the budget or narrow the layer set."
        )

    def attach(self) -> "HessianAccumulator":
        """Register the forward hooks. Raises when the set does not fit the budget."""
        import torch

        refusal = self.budget_refusal()
        if refusal:
            raise ValueError(refusal)
        for fqn, module in self.modules.items():
            k = int(module.in_features)
            device = self.device or module.weight.device
            self.hessians[fqn] = torch.zeros((k, k), dtype = torch.float32, device = device)
            self.samples[fqn] = 0
            self._handles.append(
                module.register_forward_pre_hook(self._make_hook(fqn), with_kwargs = False)
            )
        return self

    def _make_hook(self, fqn: str):
        def hook(module, args):
            if not self.active or not args:
                return None
            x = args[0]
            if x is None:
                return None
            flat = x.detach().reshape(-1, int(module.in_features)).float()
            if flat.shape[0] == 0:
                return None
            self.hessians[fqn] += flat.transpose(0, 1) @ flat
            self.samples[fqn] += int(flat.shape[0])
            return None

        return hook

    def detach(self) -> "HessianAccumulator":
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001 - an already-removed hook is fine
                pass
        self._handles = []
        self.active = False
        return self

    def free(self) -> None:
        self.hessians.clear()
        self.samples.clear()

    def normalised(self) -> dict:
        """``{fqn: H / rows}``, divided per layer, since rows differ under an attention trim."""
        out: dict = {}
        for fqn, hessian in self.hessians.items():
            rows = max(1, int(self.samples.get(fqn, 0)))
            out[fqn] = hessian / rows
        return out

    def unseen(self) -> list:
        """Layers no sampled forward reached: a Hessian of zeros corrects nothing."""
        return sorted(fqn for fqn, count in self.samples.items() if not count)


    def step_callback(self, steps: Sequence):
        """A ``callback_on_step_end`` that arms the hooks for ``steps``. It fires AFTER step i, so
        it arms step i + 1 and ``arm_first`` covers step 0."""
        wanted = {int(step) for step in steps}

        def callback(pipe, step_index, timestep, callback_kwargs):
            self.steps_seen = max(self.steps_seen, int(step_index) + 1)
            self.active = (int(step_index) + 1) in wanted
            return callback_kwargs

        return callback

    def arm_first(self, steps: Sequence) -> None:
        self.active = 0 in {int(step) for step in steps}


def sampled_steps(total_steps: int, spec: Sequence) -> tuple:
    """Which step indices to accumulate on: a short schedule is sampled at the same PLACES."""
    total = max(1, int(total_steps))
    wanted = [int(step) for step in spec]
    if wanted and max(wanted) < total:
        return tuple(sorted({min(step, total - 1) for step in wanted if step >= 0}))
    quarters = (0, total // 4, total // 2, (3 * total) // 4)
    return tuple(sorted({min(max(0, step), total - 1) for step in quarters}))


def torchao_roundtrip(weight, *, quantize_config: Any = None):
    """``weight`` through TORCHAO's quantiser and back: the scorer must measure the shipped bytes."""
    import torch
    from torchao.quantization import quantize_

    if quantize_config is None:
        from .diffusion_transformer_quant import TQ_NVFP4, _make_quant_config
        quantize_config = _make_quant_config(TQ_NVFP4)
    linear = torch.nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias = False,
        device = weight.device,
        dtype = weight.dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
    quantize_(linear, quantize_config, filter_fn = lambda module, fqn = "": True)
    packed = linear.weight
    return (
        packed.dequantize(torch.float32)
        if hasattr(packed, "dequantize")
        else packed.detach().float()
    )


def hessian_weighted_error(delta, hessian) -> float:
    """``sqrt(sum(dW H dW^T))``, the quantity GPTQ minimises. Frobenius weight error is the WRONG
    scorer: GPTQ raises it by construction."""
    import torch

    d = delta.float()
    return float(torch.sqrt(torch.clamp((d @ hessian.float() * d).sum(), min = 0.0)))


def score_correction(
    weight,
    corrected,
    hessian,
    *,
    quantize_config: Any = None,
) -> dict:
    """Hessian-weighted output error of the RTN and corrected weights, re-rounded by torchao."""
    reference = weight.detach().float()
    err_rtn = hessian_weighted_error(
        reference - torchao_roundtrip(weight, quantize_config = quantize_config), hessian
    )
    err_gptq = hessian_weighted_error(
        reference - torchao_roundtrip(corrected, quantize_config = quantize_config), hessian
    )
    return {
        "err_rtn": err_rtn,
        "err_gptq": err_gptq,
        "ratio": (err_gptq / err_rtn) if err_rtn else None,
        "improved": bool(err_gptq < err_rtn),
    }


def plan_corrections(scores: Mapping, *, max_regressions: int = 0) -> dict:
    """Which scored layers take their correction: every one that improved, plus at most
    ``max_regressions`` of the rest, least harmful first."""
    improved = sorted(fqn for fqn, score in scores.items() if score.get("improved"))
    regressed = sorted(
        (fqn for fqn, score in scores.items() if not score.get("improved")),
        key = lambda fqn: (scores[fqn].get("ratio") is None, scores[fqn].get("ratio") or 0.0),
    )
    allowed = regressed[: max(0, int(max_regressions))]
    return {
        "apply": sorted(improved + allowed),
        "improved": improved,
        "regressed": regressed,
        "applied_regressions": sorted(allowed),
        "counts": {
            "applied": len(improved) + len(allowed),
            "applied_regressed": len(allowed),
            "skipped_no_gain": len(regressed) - len(allowed),
        },
    }


class ActivationAmaxAccumulator:
    """Running ``max(abs(x))`` per layer over EVERY step, as a 0-d device tensor: the largest
    activation is the step a sampled subset would miss."""

    def __init__(self, modules: Mapping) -> None:
        self.modules = dict(modules)
        self.amax: dict = {}
        self.seen: dict = {}
        self._handles: list = []

    def attach(self) -> "ActivationAmaxAccumulator":
        import torch
        for fqn, module in self.modules.items():
            device = module.weight.device
            self.amax[fqn] = torch.zeros((), dtype = torch.float32, device = device)
            self.seen[fqn] = 0
            self._handles.append(module.register_forward_pre_hook(self._make_hook(fqn)))
        return self

    def _make_hook(self, fqn: str):
        import torch
        def hook(module, args):
            if not args or args[0] is None:
                return None
            x = args[0].detach()
            if x.numel() == 0:
                return None
            value = x.abs().amax().float()
            # A non-finite forward must not become the scale: that is the black-frame latch.
            value = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
            self.amax[fqn] = torch.maximum(self.amax[fqn], value)
            self.seen[fqn] += 1
            return None

        return hook

    def detach(self) -> "ActivationAmaxAccumulator":
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001
                pass
        self._handles = []
        return self

    def unseen(self) -> list:
        return sorted(fqn for fqn, count in self.seen.items() if not count)

    def global_scales(self) -> dict:
        """``{fqn: 6 * 448 / amax}``, the FlashInfer activation global scale, as plain floats. A
        zero or non-finite amax gets no entry, so the loader keeps that artifact on torchao."""
        import math

        out: dict = {}
        for fqn, tensor in self.amax.items():
            amax = float(tensor)
            if not math.isfinite(amax) or amax <= 0.0:
                continue
            out[fqn] = float(FP4_MAX * FP8_MAX / amax)
        return out
