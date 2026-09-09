# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The FlashInfer NVFP4 Linear, and the conversion that puts a hosted checkpoint onto it.

ONE artifact, two backends. A pre-quantized NVFP4 checkpoint is stored in torchao's payload format
and nothing here requantizes it: ``NVFP4Tensor``'s ``qdata`` IS FlashInfer's packed e2m1x2 weight
and its swizzled ``scale`` IS FlashInfer's 128x4 block-scale buffer, byte for byte, so the
conversion is a re-expression of the same bytes under different names:

    qdata            -> wq       [N, K / 2] uint8, unchanged
    scale (swizzled) -> w_sf     the same flat buffer, viewed as FlashInfer's (-1, K // 16) matrix
    per_tensor_scale == 1 / w_gsf
    alpha            =  per_tensor_scale / a_gsf

That equality is a go/no-go CUDA test (T-CUDA-3) rather than an assumption, and the unswizzled
branch below reswizzles through ``diffusion_nvfp4_ops`` for a checkpoint built without swizzling.

**The activation global scale is BAKED, never calibrated here.** The canon layer this is ported
from learned ``a_gsf`` over the first few forwards, keeping a running minimum. That state machine
is not capture-safe (its first calls mutate a buffer inside what may already be a captured graph),
it is nondeterministic (the value depends on which frames arrived first), and it is the mechanism
behind the flux black-frame latch: one non-finite forward at number 8 was frozen into the scale and
every later render was black. So the scale is measured at build time, stored per fqn in the
checkpoint metadata, and a layer whose scale is absent is NOT converted -- the artifact stays on
torchao rather than running on a guessed scale.

Everything else the research layer carried is dropped on purpose: no GFLOP routing, no fp8 weight
replica, no SmoothQuant migration, no rotation, no fused-bias nvcc extension. What is left is a
weight, a scale, an alpha and a bias.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Optional

from .diffusion_nvfp4_ops import (
    BACKEND_FLASHINFER,
    DEFAULT_MM_BACKEND,
    _device_guard,
    register_ops,
    sf_matrix_shape,
    swizzle_sf,
)

# Where the builder records the per-fqn activation global scale (``6 * 448 / act_amax``, the same
# convention FlashInfer's ``nvfp4_quantize`` takes) and the flag that says it did.
ACT_SCALES_KEY = "act_global_scales"
POLICY_KEY = "nvfp4_policy"
POLICY_BAKED_KEY = "activation_scales_baked"

# GEMM shapes already profiled, keyed ``(M, K, N)`` and shared across ALL instances. A DiT has
# hundreds of layers but only a handful of distinct shapes, and FlashInfer caches the chosen tactic
# per shape internally, so a per-module cache would pay the ~0.4 s profiling pass hundreds of times
# over for the same tactic.
_TUNED_SHAPES: set = set()


def reset_tuned_shapes() -> None:
    """Forget which GEMM shapes were autotuned. For tests and for a model unload."""
    _TUNED_SHAPES.clear()


def reset_nvfp4_state() -> None:
    """Drop every piece of process-wide NVFP4 state a loaded model left behind.

    One entry point rather than three call sites, because the thing that goes wrong here is
    forgetting one of them. Called from the unload paths next to the CUDA graph teardown, and for
    the same reason: the PDL barrier is allocated outside any capture and must not be inherited by
    the next model's graph pool. Everything reset here is cheap to rebuild, so a spurious reset
    costs a warm-up and a missed one costs a pointer into a freed pool.
    """
    from . import diffusion_nvfp4_ops as _ops

    reset_tuned_shapes()
    _ops.reset_barriers()


@lru_cache(maxsize = 1)
def nvfp4_linear_class():
    """The ``NVFP4FlashInferLinear`` class, defined on first use so this module imports torch-free.

    Cached, so the class object is a singleton and ``isinstance`` works across call sites.
    """
    import torch
    from torch import nn

    class NVFP4FlashInferLinear(nn.Module):
        """A Linear whose weight is already NVFP4 and whose activation is quantized per call.

        The forward is deliberately dull: two opaque ops and an in-place bias add. No host
        synchronize, no Python branch on a device value, and no allocation beyond what the two ops
        return, because this runs inside a captured CUDA graph.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            wq,
            w_sf,
            alpha,
            a_gsf,
            bias = None,
            backend: str = DEFAULT_MM_BACKEND,
        ):
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.backend = str(backend)
            self.register_buffer("wq", wq)
            self.register_buffer("w_sf", w_sf)
            self.register_buffer("alpha", alpha)
            self.register_buffer("a_gsf", a_gsf)
            self.register_buffer("bias", bias)
            self._tuned = False

        def forward(self, x):
            shape = x.shape
            flat = x.reshape(-1, self.in_features)
            if flat.shape[0] == 0:
                # An attention trim can hand a quantized Linear an empty batch. The GEMM has
                # nothing to compute and FlashInfer has no shape for it; a shape check is a host
                # value, not a device one, so this costs no synchronize.
                return flat.new_zeros((0, self.out_features)).reshape(
                    *shape[:-1], self.out_features
                )
            # The guard stays under torch.compile. It is a live context manager inside a traced
            # region, which is a plausible graph break, so it was measured: a two-layer NVFP4 block
            # compiles fullgraph to ONE graph with zero breaks on torch 2.12
            # (test_a_two_layer_block_compiles_fullgraph). Free, and the alternative is a launch
            # that can reach the card the process is not currently on.
            with _device_guard(flat):
                xq, x_sf = torch.ops.unsloth_nvfp4.quantize(flat, self.a_gsf)
                out = torch.ops.unsloth_nvfp4.mm(
                    xq, self.wq, x_sf, self.w_sf, self.alpha, self.out_features, self.backend
                )
            if self.bias is not None:
                # mm_fp4 has no bias epilogue (no bias argument, no beta accumulate), so the add is
                # a separate pass over the M x N output. In place, on the op's own fresh output.
                out.add_(self.bias)
            return out.reshape(*shape[:-1], self.out_features)

        def extra_repr(self) -> str:  # pragma: no cover - debug aid
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, backend={self.backend}, nvfp4=flashinfer"
            )

    return NVFP4FlashInferLinear


def is_nvfp4_flashinfer_linear(module: Any) -> bool:
    """True when ``module`` already runs the FlashInfer NVFP4 path."""
    return type(module).__name__ == "NVFP4FlashInferLinear" and hasattr(module, "a_gsf")


def is_nvfp4_tensor(t: Any) -> bool:
    """True for torchao's ``NVFP4Tensor`` without importing torchao to ask."""
    return type(t).__name__ == "NVFP4Tensor" and hasattr(t, "qdata") and hasattr(t, "scale")


def _as_scale_tensor(value: Any, *, device, dtype):
    """A 1-element fp32 tensor from a float, a 0-d tensor or a 1-element tensor."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to(device = device, dtype = dtype).reshape(1).clone()
    return torch.tensor([float(value)], device = device, dtype = dtype)


def nvfp4_linear_from_torchao(
    linear: Any,
    a_gsf: Any,
    *,
    backend: str = DEFAULT_MM_BACKEND,
):
    """Re-express one torchao NVFP4 ``nn.Linear`` as an ``NVFP4FlashInferLinear``.

    A re-expression, NEVER a requantization: every 4-bit code and every block scale in the returned
    module is the byte that was in the checkpoint. ``a_gsf`` is the baked activation global scale
    (``6 * 448 / act_amax``); the alpha the GEMM wants is ``per_tensor_scale / a_gsf``, since
    FlashInfer's alpha is ``1 / (a_gsf * w_gsf)`` and torchao's ``per_tensor_scale`` is ``1 / w_gsf``.
    """
    import torch

    register_ops()
    weight = linear.weight
    if not is_nvfp4_tensor(weight):
        raise TypeError(f"expected a torchao NVFP4Tensor weight, got {type(weight).__name__}")
    per_tensor_scale = getattr(weight, "per_tensor_scale", None)
    if per_tensor_scale is None:
        raise ValueError(
            "this NVFP4 weight carries no per_tensor_scale, so the GEMM alpha is undefined; the "
            "checkpoint was built with single-level scaling and cannot run on the flashinfer path"
        )

    qdata = weight.qdata
    wq = qdata if qdata.dtype == torch.uint8 else qdata.view(torch.uint8)
    wq = wq.contiguous()
    out_features, half_k = wq.shape[-2], wq.shape[-1]
    in_features = half_k * 2
    cols = in_features // 16

    scale = weight.scale
    if getattr(weight, "is_swizzled_scales", False):
        # torchao's ``to_blocked`` 128x4 layout and FlashInfer's ``do_shuffle = False`` buffer are
        # the same bytes in the same order (T-CUDA-3 asserts it), so this is a view, not a repack.
        flat = scale.reshape(-1)
        w_sf = flat if flat.dtype == torch.uint8 else flat.view(torch.uint8)
    else:
        w_sf = swizzle_sf(scale, out_features, in_features)
    w_sf = w_sf.reshape(sf_matrix_shape(out_features, cols)).contiguous()

    device = wq.device
    alpha = _as_scale_tensor(
        per_tensor_scale, device = device, dtype = torch.float32
    ) / _as_scale_tensor(a_gsf, device = device, dtype = torch.float32)
    bias = None if linear.bias is None else linear.bias.detach().clone()
    return nvfp4_linear_class()(
        in_features,
        out_features,
        wq = wq,
        w_sf = w_sf,
        alpha = alpha,
        a_gsf = _as_scale_tensor(a_gsf, device = device, dtype = torch.float32),
        bias = bias,
        backend = backend,
    )


def _baked_activation_scales(metadata: Any) -> Optional[dict]:
    """The per-fqn baked activation scales, or None when the artifact declares none.

    Keyed on the SCALES being present rather than on the flag alone: a checkpoint that says it
    baked them and then ships none is refused exactly like one that never claimed to.
    """
    if not isinstance(metadata, dict):
        return None
    scales = metadata.get(ACT_SCALES_KEY)
    if not isinstance(scales, dict) or not scales:
        return None
    return scales


def _declares_baked_scales(metadata: Any) -> bool:
    policy = metadata.get(POLICY_KEY) if isinstance(metadata, dict) else None
    return bool(isinstance(policy, dict) and policy.get(POLICY_BAKED_KEY))


def convert_nvfp4_backend(
    transformer: Any,
    metadata: Any,
    backend: str,
    *,
    logger: Any = None,
) -> int:
    """Move every NVFP4 Linear in ``transformer`` onto the FlashInfer path. Returns how many.

    All or nothing. A model with some layers on FlashInfer and some on torchao is a model nothing
    measured, so the walk collects first and converts only when EVERY NVFP4 Linear has a baked
    activation scale; otherwise it logs why and returns 0 with the module tree untouched, and the
    artifact runs on torchao exactly as PR 1 shipped it.
    """
    if backend != BACKEND_FLASHINFER:
        return 0
    scales = _baked_activation_scales(metadata)
    candidates = [
        (name, mod)
        for name, mod in _iter_linears(transformer)
        if is_nvfp4_tensor(getattr(mod, "weight", None))
    ]
    if not candidates:
        return 0
    if scales is None:
        _log(
            logger,
            "info",
            "[nvfp4] flashinfer backend requested but this checkpoint bakes no activation "
            f"scales ({'flag set, scales missing' if _declares_baked_scales(metadata) else 'no ' + ACT_SCALES_KEY} "
            f"for {len(candidates)} quantized linears); staying on torchao",
        )
        return 0
    missing = [name for name, _ in candidates if name not in scales]
    if missing:
        _log(
            logger,
            "info",
            f"[nvfp4] flashinfer backend refused: {len(missing)} of {len(candidates)} quantized "
            f"linears have no baked activation scale (first: {missing[0]}); staying on torchao",
        )
        return 0

    register_ops()
    converted = 0
    for name, module in candidates:
        try:
            replacement = nvfp4_linear_from_torchao(module, scales[name])
        except Exception as exc:  # noqa: BLE001 - one unconvertible layer must not lose the model
            _log(
                logger,
                "warning",
                f"[nvfp4] {name} could not move to the flashinfer path ({type(exc).__name__}: "
                f"{exc}); staying on torchao",
            )
            return converted
        _replace_child(transformer, name, replacement)
        converted += 1
    _log(logger, "info", f"[nvfp4] flashinfer backend: {converted} linears converted")
    return converted


def nvfp4_prewarm(
    transformer: Any,
    shapes: Iterable[int],
    *,
    logger: Any = None,
) -> int:
    """Autotune every converted layer at the given token counts. Returns shapes tuned.

    ``mm_fp4`` always consults FlashInfer's AutoTuner, but outside a tuning context it returns a
    DEFAULT tactic rather than a profiled one, and the default is not the best tile for diffusion
    shapes (measured 1.11x to 1.14x, bit-identical output either way). Tuning costs one profiling
    pass per distinct ``(M, K, N)``, so it belongs here -- outside the request path, and before a
    capture, since an untuned layer under capture bakes the default tactic into the graph.
    """
    import torch

    try:
        import flashinfer
    except Exception as exc:  # noqa: BLE001 - nothing to tune without the backend
        _log(logger, "debug", f"[nvfp4] prewarm skipped: {type(exc).__name__}: {exc}")
        return 0

    register_ops()
    tuned = 0
    modules = [mod for _, mod in _iter_linears(transformer) if is_nvfp4_flashinfer_linear(mod)]
    for module in modules:
        for m in shapes:
            m = int(m)
            if m <= 0:
                continue
            key = (m, module.in_features, module.out_features)
            if key in _TUNED_SHAPES:
                continue
            device = module.wq.device
            try:
                with torch.inference_mode(), _device_guard(module.wq):
                    x = torch.zeros(m, module.in_features, device = device, dtype = torch.bfloat16)
                    with flashinfer.autotune(True):
                        module(x)
                    torch.cuda.synchronize(device)
            except Exception as exc:  # noqa: BLE001 - a shape that will not tune still runs
                _log(logger, "debug", f"[nvfp4] prewarm {key} failed ({type(exc).__name__}: {exc})")
                continue
            _TUNED_SHAPES.add(key)
            tuned += 1
        module._tuned = True
    if tuned:
        _log(logger, "info", f"[nvfp4] prewarm tuned {tuned} GEMM shapes")
    return tuned


# ── module-tree plumbing ──────────────────────────────────────────────────────────────────────


def _iter_linears(transformer: Any):
    """``(fqn, module)`` for every leaf that looks like a Linear, in definition order.

    Keyed on the two feature counts rather than on ``weight``, so that the walk still finds a layer
    this module has ALREADY converted -- an ``NVFP4FlashInferLinear`` holds packed buffers and no
    ``weight`` at all, and the prewarm has to be able to reach it.
    """
    for name, module in transformer.named_modules():
        if not name:
            continue
        if hasattr(module, "in_features") and hasattr(module, "out_features"):
            yield name, module


def _replace_child(root: Any, fqn: str, replacement: Any) -> None:
    parent = root
    parts = fqn.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
    last = parts[-1]
    if last.isdigit() and hasattr(parent, "__setitem__"):
        parent[int(last)] = replacement
    else:
        setattr(parent, last, replacement)


def _log(logger: Any, level: str, message: str) -> None:
    if logger is None:
        return
    try:
        getattr(logger, level)(message)
    except Exception:  # noqa: BLE001 - logging is best effort
        pass
