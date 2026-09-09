# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GPTQ correction and activation-scale baking for the NVFP4 layers of a prequant build.

Two build-time passes over a DENSE pipeline, both driven by the same calibration prompts and both
scoped to the layers a build actually quantises to 4 bits (a policy's NVFP4 set, or every admitted
linear for a whole-model video artifact):

  * **Hessians and GPTQ.** Round-to-nearest is the weakest possible quantiser. GPTQ corrects the
    weight onto the same NVFP4 grid using the layer's own input second-moment matrix, which on real
    z-image weights takes the 4-bit weight error from 0.0215 to 0.0064 against fp8's 0.0067. The
    correction is applied PER LAYER and only where it is measured to help, through torchao's own
    quantiser, because a correction that is right on paper and wrong on this model is a silently
    worse artifact.
  * **Activation global scales.** The FlashInfer NVFP4 layer needs one ``a_gsf = 6 * 448 /
    act_amax`` per layer. Measuring it at run time is what the shipped canon layer did, and it is
    not capture-safe, not deterministic, and the mechanism behind the flux black-frame latch (one
    non-finite forward frozen into a running minimum). So it is measured here, once, on the
    calibration set, and stored in the checkpoint.

Scope, deliberately: this module accumulates, corrects and scores. It does not decide which layers
to touch (the policy does), it does not run the pipeline (the builder does) and it never falls back
to a different quantiser -- ``gptq_correct`` raises when its Cholesky will not factor at any damping
in the ladder, because "the correction failed, so the weight is plain RTN" is a different artifact
than the one that was asked for and nothing would say so.

The two quantisers below are ported from ``scripts/nvfp4_linear_canon.py`` (the shipping candidate
of the research layer) and kept verbatim: they define the grid the correction lands on, and the
whole point of correcting onto the grid is that torchao's re-rounding at build time reproduces it.

torch is imported inside the functions, like the other lazily loaded inference helpers.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

# The NVFP4 grid: e2m1 elements up to 6.0, an e4m3 per-block scale up to 448.0, one fp32 global
# scale. The activation global scale a layer stores is the same convention FlashInfer's
# ``nvfp4_quantize`` takes, ``6 * 448 / amax``.
FP4_MAX = 6.0
FP8_MAX = 448.0

# Damping ladder for the Cholesky, smallest first. A Hessian accumulated over a few hundred
# forwards of a 3072-wide layer is positive SEMI-definite at best, so the first factorisation can
# fail; the answer is more damping, not another quantiser.
DAMP_LADDER: tuple = (0.01, 0.05, 0.1, 0.5, 1.0)


class GPTQFailure(RuntimeError):
    """The correction could not be solved for a layer. Raised, never swallowed.

    Falling back to round-to-nearest here would produce an artifact that is PARTLY corrected with
    nothing in the metadata to say which half, which is exactly the state the do-no-harm scoring
    exists to keep out of a checkpoint."""


def gptq_quantize_to_nvfp4(
    weight,
    hessian,
    block: int = 16,
    damp: float = 0.01,
):
    """GPTQ-correct a weight ONTO the NVFP4 grid. Returns a bf16 tensor already representable.

    Round-to-nearest is the weakest possible quantiser and it is what every 4-bit number in this
    investigation used. Measured on real z-image weights with real activations and a HELD-OUT
    evaluation split, this takes 4-bit weight error from 0.0215 to 0.0064 against fp8's 0.0067,
    i.e. from 3.2x worse than fp8 to parity, at 4.5 bits/weight instead of 8.

    The output must land exactly on the format's grid (e2m1 values, fp8 per-block scale, one global
    scale), because it is then handed to the quantiser that packs the checkpoint. If the grid here
    disagreed with that one's, the requantisation would silently undo the correction.
    """
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
        # fp8-rounded block scale: the format stores it in e4m3, so rounding it here is what makes
        # the result exactly representable downstream.
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
    """``(corrected_weight, damp)``, escalating the damping until the Cholesky factors.

    Escalation, never substitution: each rung is a slightly more regularised version of the SAME
    correction, so the artifact stays the thing the build asked for and the damping that produced
    it is recorded per layer. When the last rung fails the layer raises ``GPTQFailure`` and the
    build stops, rather than quietly shipping this one weight at round-to-nearest."""
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


# ── Hessians ──────────────────────────────────────────────────────────────────────────────────


class HessianAccumulator:
    """Per-layer ``sum(x^T x)`` in fp32 over the sampled steps of a calibration run.

    Bounded on purpose. One K x K fp32 matrix per layer is the whole memory cost of the pass and it
    is quadratic in the input width, so the budget is checked UP FRONT against the layers that were
    asked for and the pass refuses rather than filling the card halfway through a 40 minute
    calibration. On the sets this is used for (z-image's 34 to_q at K = 3072, qwen's 120 modulation
    projections at K = 3072) the cost is 1.3 to 4.5 GB.

    Accumulation is gated by ``active`` rather than by attaching and detaching hooks per step: the
    pipeline calls the denoiser twice per step under real CFG, and a flag catches both halves of a
    sampled step without assuming anything about how many forwards a step makes.
    """

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
        """``{fqn: H / rows}``, the mean second moment. Rows differ per layer under an attention
        trim, so the division is per layer and not one global count."""
        out: dict = {}
        for fqn, hessian in self.hessians.items():
            rows = max(1, int(self.samples.get(fqn, 0)))
            out[fqn] = hessian / rows
        return out

    def unseen(self) -> list:
        """Layers no sampled forward ever reached. A Hessian of zeros corrects nothing, and a
        layer the calibration never exercised is a layer this build cannot claim to have measured."""
        return sorted(fqn for fqn, count in self.samples.items() if not count)

    # ``callback_on_step_end`` support ---------------------------------------------------------

    def step_callback(self, steps: Sequence):
        """A diffusers ``callback_on_step_end`` that arms the hooks for ``steps`` only.

        The callback fires AFTER step i, so it arms step i + 1; step 0 is armed by ``arm_first``
        before the pipeline is called. Returning the kwargs unchanged is the contract."""
        wanted = {int(step) for step in steps}

        def callback(pipe, step_index, timestep, callback_kwargs):
            self.steps_seen = max(self.steps_seen, int(step_index) + 1)
            self.active = (int(step_index) + 1) in wanted
            return callback_kwargs

        return callback

    def arm_first(self, steps: Sequence) -> None:
        self.active = 0 in {int(step) for step in steps}


def sampled_steps(total_steps: int, spec: Sequence) -> tuple:
    """Which step indices to accumulate on, for a schedule of ``total_steps``.

    ``spec`` is the default 38-step schedule's sample points (0, 12, 25, 37: start, both thirds and
    the end). A distilled model runs 4 to 8 steps, where those indices do not exist, so a shorter
    schedule is sampled at the same PLACES rather than the same indices: 0, n/4, n/2, 3n/4. The
    result is deduplicated and clamped, so a 1-step schedule samples step 0 and nothing else."""
    total = max(1, int(total_steps))
    wanted = [int(step) for step in spec]
    if wanted and max(wanted) < total:
        return tuple(sorted({min(step, total - 1) for step in wanted if step >= 0}))
    quarters = (0, total // 4, total // 2, (3 * total) // 4)
    return tuple(sorted({min(max(0, step), total - 1) for step in quarters}))


# ── do no harm ────────────────────────────────────────────────────────────────────────────────


def torchao_roundtrip(weight, *, quantize_config: Any = None):
    """``weight`` through TORCHAO's own NVFP4 quantiser and back to fp32.

    The scorer has to measure the error of the weight that ends up IN the checkpoint, and that
    weight is whatever ``quantize_`` packs. Scoring the grid maths in this module instead would
    measure a quantiser the artifact does not use, which is how a correction that torchao re-rounds
    away gets shipped as an improvement."""
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
    """``sqrt(sum(dW H dW^T))``: the output error a weight perturbation causes on the calibration
    activations, which is the quantity GPTQ minimises. The plain Frobenius weight error is the
    wrong scorer here -- GPTQ RAISES it by construction, trading it for this one."""
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
    """Hessian-weighted output error of the RTN weight and of the corrected one, re-rounded by
    torchao. ``improved`` is the do-no-harm verdict for this one layer."""
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
    ``max_regressions`` of the rest, least harmful first.

    Do no harm is the default and the whole point: a correction is applied only where it is
    MEASURED to lower this layer's output error through the quantiser the build will use.
    ``--gptq-max-regressions`` above 0 is the deliberate escape hatch for reproducing a campaign
    that applied a whole set, and it applies the least harmful ones first so the number means
    something."""
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


# ── activation global scales ──────────────────────────────────────────────────────────────────


class ActivationAmaxAccumulator:
    """Running ``max(abs(x))`` per layer, over EVERY step of every calibration prompt.

    Every step, not the sampled ones: the scale has to cover the whole trajectory, and the step
    that produces the largest activation is exactly the one a sampled subset would miss. Kept on
    the device as a 0-d tensor per layer so a 30-step render costs no host synchronise."""

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
            # A non-finite forward must not become the scale every later render is quantised by:
            # that is the flux black-frame latch, and baking it would make it permanent.
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
        """``{fqn: 6 * 448 / amax}``, the FlashInfer activation global scale, as plain floats.

        A layer whose amax is zero or non-finite gets no entry rather than an infinite scale: the
        loader refuses to convert an artifact with a missing scale and runs it on torchao, which is
        the right outcome for a layer this calibration never measured."""
        import math

        out: dict = {}
        for fqn, tensor in self.amax.items():
            amax = float(tensor)
            if not math.isfinite(amax) or amax <= 0.0:
                continue
            out[fqn] = float(FP4_MAX * FP8_MAX / amax)
        return out
