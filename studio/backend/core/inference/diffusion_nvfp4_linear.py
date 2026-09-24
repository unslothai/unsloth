# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The FlashInfer NVFP4 Linear, and the conversion that puts a hosted checkpoint onto it.

ONE artifact, two backends. Nothing here requantizes: torchao's payload IS FlashInfer's, byte for
byte. The activation global scale is BAKED, never calibrated (not capture-safe, and the cause of the
flux black-frame latch), so a layer without one is NOT converted.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Optional

from .diffusion_nvfp4_ops import (
    BACKEND_FLASHINFER,
    DEFAULT_MM_BACKEND,
    _device_guard,
    dequantize_nvfp4_weight,
    register_ops,
    sf_matrix_shape,
    swizzle_sf,
)
from .diffusion_nvfp4_protect import protect_controller

ACT_SCALES_KEY = "act_global_scales"
POLICY_KEY = "nvfp4_policy"
POLICY_BAKED_KEY = "activation_scales_baked"

_TUNED_SHAPES: set = set()


def reset_tuned_shapes() -> None:
    _TUNED_SHAPES.clear()


def reset_nvfp4_state() -> None:
    from . import diffusion_nvfp4_dispatch as _dispatch
    from . import diffusion_nvfp4_ops as _ops

    reset_tuned_shapes()
    _ops.reset_barriers()
    _dispatch.reset()
    # verify() runs only in the preflight, so a memoised one would leave _VERIFIED empty next load.
    _ops.reset_preflight_cache()


@lru_cache(maxsize = 1)
def nvfp4_linear_class():
    import torch
    from torch import nn
    from torch.nn import functional as F

    from .diffusion_nvfp4_bias import fused_bias_add_

    class NVFP4FlashInferLinear(nn.Module):
        """NVFP4-weight Linear; the forward must stay capture-safe (no host sync, no device-value branch)."""

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
            w_scale = None,
            backend: str = DEFAULT_MM_BACKEND,
            activation_scales_baked: bool = False,
        ):
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.backend = str(backend)
            self.activation_scales_baked = bool(activation_scales_baked)
            self.register_buffer("wq", wq)
            self.register_buffer("w_sf", w_sf)
            self.register_buffer("alpha", alpha)
            self.register_buffer("a_gsf", a_gsf)
            # Stored, not ``alpha * a_gsf``, which is off by a rounding: W4A16 must match bit for bit.
            self.register_buffer(
                "w_scale",
                (alpha * a_gsf) if w_scale is None else w_scale,
            )
            self.register_buffer("bias", bias)
            self._tuned = False
            self.protect = protect_controller()
            self.protect.register_layer(self)

        def forward(self, x):
            shape = x.shape
            flat = x.reshape(-1, self.in_features)
            # ``fp4_quantize`` raises on fp32, which Wan2.2's time embedder feeds in.
            out_dtype = flat.dtype
            if out_dtype not in (torch.bfloat16, torch.float16):
                flat = flat.to(torch.bfloat16)
            if flat.shape[0] == 0:
                return flat.new_zeros((0, self.out_features), dtype = out_dtype).reshape(
                    *shape[:-1], self.out_features
                )
            if self.protect.armed and self.protect.protected:
                weight = dequantize_nvfp4_weight(
                    self.wq, self.w_sf, self.w_scale, dtype = torch.bfloat16
                )
                out = F.linear(flat, weight)
            else:
                with _device_guard(flat):
                    xq, x_sf = torch.ops.unsloth_nvfp4.quantize(flat, self.a_gsf)
                    out = torch.ops.unsloth_nvfp4.mm(
                        xq, self.wq, x_sf, self.w_sf, self.alpha, self.out_features, self.backend
                    )
            out = out.to(out_dtype)
            if self.bias is not None:
                # mm_fp4 has no bias epilogue, so the add is a separate pass.
                fused_bias_add_(out, self.bias)
            return out.reshape(*shape[:-1], self.out_features)

        def extra_repr(self) -> str:  # pragma: no cover - debug aid
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, backend={self.backend}, nvfp4=flashinfer"
                + (f", protect={self.protect.spec}" if self.protect.armed else "")
            )

    return NVFP4FlashInferLinear


def is_nvfp4_flashinfer_linear(module: Any) -> bool:
    return type(module).__name__ == "NVFP4FlashInferLinear" and hasattr(module, "a_gsf")


def is_nvfp4_tensor(t: Any) -> bool:
    return type(t).__name__ == "NVFP4Tensor" and hasattr(t, "qdata") and hasattr(t, "scale")


def _as_scale_tensor(value: Any, *, device, dtype):
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
        # Kept as torchao stored it: the W4A16 branch must land on torchao's own bytes.
        w_scale = _as_scale_tensor(per_tensor_scale, device = device, dtype = torch.float32),
        backend = backend,
        activation_scales_baked = True,
    )


def _baked_activation_scales(metadata: Any) -> Optional[dict]:
    if not isinstance(metadata, dict):
        return None
    scales = metadata.get(ACT_SCALES_KEY)
    if not isinstance(scales, dict) or not scales:
        return None
    return scales


def _declares_baked_scales(metadata: Any) -> bool:
    if not isinstance(metadata, dict):
        return False
    policy = metadata.get(POLICY_KEY)
    if isinstance(policy, dict) and policy.get(POLICY_BAKED_KEY):
        return True
    return bool(metadata.get(POLICY_BAKED_KEY))


def convert_nvfp4_backend(
    transformer: Any,
    metadata: Any,
    backend: str,
    *,
    logger: Any = None,
) -> int:
    """Move every NVFP4 Linear onto FlashInfer, returning how many; all or nothing on a missing baked scale."""
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
    replacements = []
    for name, module in candidates:
        try:
            replacements.append((name, nvfp4_linear_from_torchao(module, scales[name])))
        except Exception as exc:  # noqa: BLE001 - one unconvertible layer must not lose the model
            _log(
                logger,
                "warning",
                f"[nvfp4] {name} could not move to the flashinfer path ({type(exc).__name__}: "
                f"{exc}); staying on torchao",
            )
            return 0
    for name, replacement in replacements:
        _replace_child(transformer, name, replacement)
    from .diffusion_nvfp4_protect import attach_own_controller

    attach_own_controller(transformer)
    converted = len(replacements)
    _log(logger, "info", f"[nvfp4] flashinfer backend: {converted} linears converted")
    return converted


def nvfp4_prewarm(
    transformer: Any,
    shapes: Iterable[int],
    *,
    logger: Any = None,
) -> int:
    """Autotune converted layers at these token counts BEFORE any capture, with the precision lever suspended."""
    from .diffusion_nvfp4_flag import nvfp4_diffusion_enabled

    if not nvfp4_diffusion_enabled():
        return 0
    import torch

    try:
        import flashinfer
    except Exception as exc:  # noqa: BLE001 - nothing to tune without the backend
        _log(logger, "debug", f"[nvfp4] prewarm skipped: {type(exc).__name__}: {exc}")
        return 0

    from .diffusion_nvfp4_protect import suspend_protect

    register_ops()
    modules = [mod for _, mod in _iter_linears(transformer) if is_nvfp4_flashinfer_linear(mod)]
    counter = [0]
    with suspend_protect(modules):
        _prewarm_shapes(modules, shapes, logger = logger, tuned_box = counter)
    tuned = counter[0]
    if tuned:
        _log(logger, "info", f"[nvfp4] prewarm tuned {tuned} GEMM shapes")
    return tuned


def _prewarm_shapes(modules, shapes, *, logger, tuned_box) -> None:
    import torch

    import flashinfer
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
            tuned_box[0] += 1
        module._tuned = True


def _iter_linears(transformer: Any):
    """Every Linear-like leaf, by feature counts (a converted layer has no ``weight``)."""
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
