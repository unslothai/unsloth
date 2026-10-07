# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pad an int8 Linear's activation rows up to ``torch._int_mm``'s floor, instead of skipping it.

``aten::_int_mm`` asserts ``self.size(0) > 16``. torchao's EAGER path never trips this, because
``safe_int_mm`` checks the cuBLAS dimension constraints at runtime and falls back to a widened
``torch.matmul``; inductor lowers the same quantized linear straight to ``_int_mm``, so under
``torch.compile`` any quantized Linear invoked with a small activation row count raises

    RuntimeError: self.size(0) needs to be greater than 16, but got 10

Until now the fix was to leave those Linears dense bf16 (``_INT8_FAMILY_EXCLUDE_NAME_TOKENS`` in
``diffusion_transformer_quant``). That works, but it forfeits the weight-memory saving on a DiT's
whole conditioning front end. This is the alternative: pad the flattened row count up to
``pad_to``, run the GEMM, slice the result back. The module becomes compilable with no change to
the quantization config, and the rows the caller asked for are returned BITWISE unchanged.

Exact because pad rows REPLICATE row 0 (a zero row gives scale 0 and NaN) and the activation scale
is PER ROW; ``wrap_small_m_linears`` RAISES on a granularity it cannot prove per row, since a
half-padded transformer crashes. ``ZeroRowSafeLinear`` answers an EMPTY activation itself (nvfp4).
Wrapping reparents the Linear, so do it AFTER loading; ``PadToMinM`` is state-dict transparent.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
from torch import nn

# ``_int_mm`` wants strictly more than 16 rows.
INT_MM_MIN_M = 17

# Pad to 32 (not 17): better tiling and one compiled shape for all smaller M.
DEFAULT_PAD_TO = 32


def _weight_tensors(module: Any) -> tuple:
    """The weight objects a granularity/layout probe should look at, most specific first.

    ``quantize_`` assigns ``nn.Parameter(subclass_tensor)``, and Parameter.__new__ returns the
    SUBCLASS itself for a non-plain tensor, so the attributes normally sit on ``weight``. A build
    that produced a real Parameter wrapper instead would hide them one level down, hence ``.data``
    as a second look."""
    weight = getattr(module, "weight", None)
    if weight is None:
        return ()
    data = getattr(weight, "data", None)
    return (weight,) if data is None or data is weight else (weight, data)


def is_quantized_linear(module: Any) -> bool:
    """True when ``module``'s weight is a torchao tensor subclass rather than a plain tensor.

    A dense Linear needs no padding (``F.linear`` has no row floor), so callers use this to skip
    one rather than to fail on it."""
    if not isinstance(module, nn.Linear):
        return False
    return any(hasattr(t, "__tensor_flatten__") for t in _weight_tensors(module))


def activation_granularity_is_per_row(module: Any) -> Optional[bool]:
    """Whether ``module`` quantizes ACTIVATIONS per row. None when it cannot be determined.

    torchao spells this two ways depending on the tensor generation, and neither is a public
    accessor, so both are probed and an unrecognised layout answers None (the caller treats that
    as "unproven" and refuses, rather than assuming):

      * v2 tensors (``Float8Tensor`` and friends) carry ``act_quant_kwargs.granularity``, which
        is a ``PerRow`` instance for the per-row configs.
      * v1 ``LinearActivationQuantizedTensor`` (what ``Int8DynamicActivationInt8WeightConfig``
        produces today) carries the activation quantizer as ``input_quant_func``; the per-row
        one is ``_int8_symm_per_token_reduced_range_quant``. Note that this holds regardless of
        the config's ``granularity=`` argument, which sets the WEIGHT granularity -- so the
        function name is the only honest signal here.
    """
    for tensor in _weight_tensors(module):
        kwargs = getattr(tensor, "act_quant_kwargs", None)
        granularity = getattr(kwargs, "granularity", None)
        if granularity is not None:
            return type(granularity).__name__ in ("PerRow", "PerToken")

        quant_fn = getattr(tensor, "input_quant_func", None)
        name = getattr(quant_fn, "__name__", "") if quant_fn is not None else ""
        if name:
            lowered = name.lower()
            if "per_token" in lowered or "per_row" in lowered:
                return True
            if "per_tensor" in lowered:
                return False
    return None


class PadToMinM(nn.Module):
    """Wrap ``inner`` so its GEMM never sees fewer than ``pad_to`` activation rows (>= ``min_m``).

    Shape-preserving: the caller's leading dims come back untouched. Only the FLATTENED row
    count is padded, and only when it is below ``pad_to``, so a module that is small on one call
    and large on the next pays nothing on the large one.
    """

    def __init__(
        self,
        inner: nn.Linear,
        min_m: int = INT_MM_MIN_M,
        pad_to: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.min_m = int(min_m)
        self.pad_to = max(int(pad_to or min_m), self.min_m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No mutable int attributes: dynamo guards on them, forcing recompiles then eager fallback.
        lead = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        m = flat.shape[0]
        if m == 0:
            # torchao returns a zero-row input unprojected (wrong width), so build the empty result.
            return x.new_empty((*lead, self.inner.out_features))
        if torch.compiler.is_compiling():
            # Branch-free: `m < pad_to` would guard a dynamic row count; dynamo <= 2.8 rejects sym_max.
            rows = torch.arange((m + self.pad_to + abs(m - self.pad_to)) // 2, device = flat.device)
            out = self.inner(flat.index_select(0, torch.where(rows < m, rows, 0)))[:m]
        elif m < self.pad_to:
            flat = torch.cat([flat, flat[:1].expand(self.pad_to - m, -1)], dim = 0)
            out = self.inner(flat)[:m]
        else:
            out = self.inner(flat)
        return out.reshape(*lead, out.shape[-1])

    def __getattr__(self, name: str) -> Any:
        # Forward attribute access to the inner Linear (callers read e.g. `to_q.weight.dtype`).
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "inner":
                raise
            inner = self._modules.get("inner")
            if inner is None:
                raise
            return getattr(inner, name)

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """Emit the inner Linear's tensors under the WRAPPER's prefix, hiding the ``inner.`` level.

        ``nn.Module.state_dict`` recurses by calling each child's ``state_dict``, so overriding it
        here is enough to keep a checkpoint written from a wrapped transformer loadable by an
        unwrapped one. The wrapper owns no tensors of its own, so there is nothing else to emit."""
        destination = kwargs.pop("destination", args[0] if args else None)
        prefix = kwargs.pop("prefix", args[1] if len(args) > 1 else "")
        keep_vars = kwargs.pop("keep_vars", args[2] if len(args) > 2 else False)
        if destination is None:
            return self.inner.state_dict(prefix = prefix, keep_vars = keep_vars)
        self.inner.state_dict(destination = destination, prefix = prefix, keep_vars = keep_vars)
        return destination

    def _load_from_state_dict(
        self,
        state_dict: Any,
        prefix: str,
        local_metadata: Any,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        """Accept the unwrapped key names ``state_dict`` above writes, and hand them to ``inner``.

        Rewrites ``<prefix>weight`` to ``<prefix>inner.weight`` in place, before
        ``nn.Module.load_state_dict``'s recursion descends into ``inner``, so a state dict saved
        from an UNWRAPPED transformer loads into a wrapped one. Keys already carrying ``inner.``
        are left alone, so a dict written by an older build still loads."""
        for key in [k for k in state_dict if k.startswith(prefix)]:
            leaf = key[len(prefix) :]
            if not leaf or leaf.startswith("inner."):
                continue
            state_dict[prefix + "inner." + leaf] = state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def extra_repr(self) -> str:
        return f"min_m = {self.min_m}, pad_to = {self.pad_to}"


class ZeroRowSafeLinear(nn.Module):
    """Answer an EMPTY activation here: torchao NVFP4's global-scale ``max()`` raises on ``numel() == 0``."""

    def __init__(self, inner: nn.Linear) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            out = x.new_zeros((*x.shape[:-1], self.inner.out_features))
            bias = getattr(self.inner, "bias", None)
            return out if bias is None else out + bias
        return self.inner(x)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "inner":
                raise
            inner = self._modules.get("inner")
            if inner is None:
                raise
            return getattr(inner, name)

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """Emit the inner Linear's tensors under the wrapper's prefix, loadable unwrapped."""
        destination = kwargs.pop("destination", args[0] if args else None)
        prefix = kwargs.pop("prefix", args[1] if len(args) > 1 else "")
        keep_vars = kwargs.pop("keep_vars", args[2] if len(args) > 2 else False)
        if destination is None:
            return self.inner.state_dict(prefix = prefix, keep_vars = keep_vars)
        self.inner.state_dict(destination = destination, prefix = prefix, keep_vars = keep_vars)
        return destination

    def _load_from_state_dict(
        self,
        state_dict: Any,
        prefix: str,
        local_metadata: Any,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        """Accept the unwrapped key names ``state_dict`` above writes, and hand them to ``inner``."""
        for key in [k for k in state_dict if k.startswith(prefix)]:
            leaf = key[len(prefix) :]
            if not leaf or leaf.startswith("inner."):
                continue
            state_dict[prefix + "inner." + leaf] = state_dict.pop(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


def wrap_zero_row_linears(model: nn.Module, fqns: Iterable[str]) -> tuple[str, ...]:
    """Wrap each Linear in ``fqns`` in a ``ZeroRowSafeLinear``; idempotent, and never stacks on ``PadToMinM``."""
    done: list[str] = []
    for fqn in sorted(set(fqns)):
        parent_name, _, leaf = fqn.rpartition(".")
        try:
            parent = model.get_submodule(parent_name) if parent_name else model
            module = getattr(parent, leaf)
        except AttributeError:
            continue
        if not is_quantized_linear(module):
            continue
        setattr(parent, leaf, ZeroRowSafeLinear(module))
        done.append(fqn)
    return tuple(done)


def padding_is_bitwise_exact(
    module: Any,
    m: int,
    *,
    pad_to: int = DEFAULT_PAD_TO,
) -> bool:
    """Run ``module`` at ``m`` rows with and without padding and report bitwise equality.

    The direct form of the property the granularity check infers. Used by the tests; too costly
    for a per-module load-time gate on a 300-Linear DiT, and it cannot replace the granularity
    check anyway (replicating row 0 leaves a per-tensor AMAX unchanged, so a per-tensor amax
    scheme would pass this while a percentile-calibrated one would not)."""
    weight = getattr(module, "weight", None)
    device = getattr(weight, "device", "cpu")
    dtype = getattr(weight, "dtype", torch.bfloat16)
    x = torch.randn(m, module.in_features, device = device, dtype = dtype)
    with torch.no_grad():
        reference = module(x)
        padded = PadToMinM(module, pad_to = pad_to)(x)
    return bool(torch.equal(reference, padded))


def wrap_small_m_linears(
    model: nn.Module,
    fqns: Iterable[str],
    *,
    min_m: int = INT_MM_MIN_M,
    pad_to: Optional[int] = DEFAULT_PAD_TO,
    require_per_row: bool = True,
) -> tuple[str, ...]:
    """Replace each Linear named in ``fqns`` with a ``PadToMinM`` around it; return those wrapped.

    Surgical by construction: only the fqns handed in are touched, so a DiT's hundreds of
    large-M block linears keep their unwrapped fast path and cannot pay for this.

    A Linear that is NOT quantized is skipped (dense ``F.linear`` has no row floor to clear), as
    is one already wrapped. A QUANTIZED Linear whose activation granularity cannot be proven per
    row raises ``RuntimeError``: see the module docstring for why silence is the wrong answer.
    """
    done: list[str] = []
    for fqn in sorted(set(fqns)):
        parent_name, _, leaf = fqn.rpartition(".")
        try:
            parent = model.get_submodule(parent_name) if parent_name else model
            module = getattr(parent, leaf)
        except AttributeError:
            continue
        # Skips dense Linears and already-wrapped PadToMinM (not an nn.Linear), so no double pad.
        if not is_quantized_linear(module):
            continue
        if require_per_row and activation_granularity_is_per_row(module) is not True:
            raise RuntimeError(
                f"{fqn}: refusing to pad a quantized Linear whose activation granularity is not "
                f"provably per row. Padding replicates row 0, which is exact only when each "
                f"kept row's scale is computed from that row alone; under a calibrated or "
                f"per-tensor activation scale it would silently change every output."
            )
        setattr(parent, leaf, PadToMinM(module, min_m = min_m, pad_to = pad_to))
        done.append(fqn)
    return tuple(done)


def matching_linear_fqns(model: nn.Module, name_tokens: Iterable[str]) -> tuple[str, ...]:
    """Every quantized-Linear fqn in ``model`` containing one of ``name_tokens`` (substring,
    case-insensitive) -- the same matching rule ``make_filter_fn`` uses for exclusions, so the
    pad list and the exclude list are read the same way."""
    tokens = tuple(t.lower() for t in name_tokens if t)
    if not tokens:
        return ()
    return tuple(
        fqn
        for fqn, module in model.named_modules()
        if is_quantized_linear(module) and any(t in fqn.lower() for t in tokens)
    )
