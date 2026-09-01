# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ConvRot: block-Hadamard activation rotation for a pre-quantized DiT transformer.

A dynamic-activation INT8 Linear quantizes BOTH sides of its GEMM to 8 bits, so its error is set
by whichever side has the heaviest outliers. A DiT's activations are outlier-heavy by
construction: a handful of channels carry most of the magnitude, the per-row amax is pinned by
them, and every other channel spends its 8 bits on a fraction of the range it could have used.

Rotating the input axis by an orthogonal matrix spreads that magnitude across the group. With a
normalized regular Hadamard ``H`` of size ``g``, applied blockwise along the input axis:

    offline:  W[o, k, :] <- W[o, k, :] @ H.T          (baked into the checkpoint)
    online:   x[..., k, :] <- x[..., k, :] @ H        (this module, at every forward)

``H`` is symmetric and orthogonal, so ``(x @ H) @ (W @ H.T).T == x @ W.T``: in float the pair is
an exact identity and the model is unchanged. What changes is only that the INT8 quantizer sees a
flatter distribution on both sides.

The arithmetic is the SAME ConvRot the hosted Qwen3-VL conditioner already runs, so
``build_convrot_hadamard`` / ``rotate_convrot_activation`` live here and
``video_minimax_h3_te`` imports them. One Hadamard in the tree, not two: the conditioner's and
the denoiser's rotations have to agree with the same comfy-kitchen definition, and two copies of
a matrix nobody re-derives at review time is how they stop agreeing.

What this module adds on top of those primitives is the DENOISER half: the offline weight
rotation, the ``nn.Linear`` subclass that applies the online half, and the metadata contract that
keeps the two in step.

Why the fqn list is RECORDED rather than recomputed
---------------------------------------------------
The two halves of the identity live in different places: one in the checkpoint's weights, one in
this process. If they ever disagree about WHICH Linears are rotated, the mismatched ones compute
``x @ H @ W.T`` or ``x @ (W @ H.T).T`` -- finite, plausible-looking, and completely wrong. There
is no exception to catch and no NaN to notice; the render is just quietly worse. So the
checkpoint carries the exact list of rotated fqns and the loader rotates that list, rather than
whatever a re-evaluated filter rule selects today. A rule can drift with a code change (the set
is "quantized Linears whose in_features the group divides", and both halves of that have moved
before); a list cannot.

Everything here fails CLOSED for the same reason. ``apply_activation_rotation`` RAISES on an fqn
it cannot find, on a module that is not a Linear, on an ``in_features`` the group does not
divide, and on a rotation kind it does not implement. Its caller (the prequant loader) turns that
into a refused checkpoint and a dense fallback, which is slow but correct. Rotated artifacts also
carry their own format tag, so an Unsloth old enough to predate this module rejects them outright
instead of running them unrotated.

torch is imported inside the functions, matching the other lazily-loaded inference helpers, so
reading the metadata contract costs no import.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Optional

# ── the metadata contract, carried in the prequant checkpoint's own ``metadata`` dict ──────────
# The rotation KIND. A value this module does not implement is refused, so a future scheme can be
# added without a released Unsloth silently treating it as this one.
CONVROT_KIND = "convrot_hadamard_v1"
# Key naming mirrors the adaLN curve contract next door (``adaln_form`` / ``curve_dim`` /
# ``curve_grid``): a form tag plus the parameters needed to reproduce the form.
ROTATION_KEY = "activation_rotation"
ROTATION_GROUP_KEY = "activation_rotation_group"
ROTATION_FQNS_KEY = "activation_rotation_fqns"

# The group size the denoiser artifact ships at, and the one the hosted conditioner already uses.
# 256 beat 64 in weight space on MiniMax-H3 (mean relative quantization error -19.9% vs -17.3%
# over 200 layers) and is the largest power of 4 that divides every quantized H3 input axis.
DEFAULT_CONVROT_GROUPSIZE = 256

# Marker set on a transformer whose rotation is installed, so a caller can tell a rotated module
# from an unrotated one without re-deriving anything. Diagnostic only.
CONVROT_ATTR = "_unsloth_activation_rotation"


def is_power_of_four(size: Any) -> bool:
    """True for 4, 16, 64, 256, ... -- the sizes a kron power of ``H4`` can produce.

    A genuine ``int`` only. ``"256"`` is not accepted even though it would coerce: this also gates
    the group recorded in a checkpoint, and a value whose TYPE is already wrong says the artifact
    was not written by the builder in this tree, which is not something to guess about."""
    if not isinstance(size, int) or isinstance(size, bool):
        return False
    n = size
    if n < 4 or n & (n - 1):
        return False
    # A power of two is a power of four exactly when its single set bit sits at an even index.
    return (n.bit_length() - 1) % 2 == 0


# ── the rotation itself ───────────────────────────────────────────────────────────────────────
# Mirrors comfy-kitchen's ``_build_hadamard`` / ``_rotate_activation`` / ``_rotate_weight``, in a
# few lines of torch rather than a dependency on a wheel Unsloth does not ship.

_HADAMARD_CACHE: dict = {}


def build_convrot_hadamard(
    size: int,
    device: Any = "cpu",
    dtype: Any = None,
) -> Any:
    """The normalized regular Hadamard matrix ConvRot rotates by. Cached per (size, device, dtype).

    Built as ``kron(H4, H4, ...) / sqrt(size)``, which is both symmetric and orthogonal -- the
    property the offline/online pair relies on, since it means the same matrix undoes itself and
    the weight side can use ``H.T`` interchangeably with ``H``. Building directly in ``dtype`` is
    exact for every float type: the entries are +-1 and the normalizer is a power of two."""
    import torch

    if dtype is None:
        dtype = torch.float32
    key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    if not is_power_of_four(size):
        raise ValueError(f"ConvRot group size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype = dtype,
        device = device,
    )
    h = h4
    current = 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    h = h / (size**0.5)
    _HADAMARD_CACHE[key] = h
    return h


def rotate_convrot_activation(x: Any, h: Any, group_size: int) -> Any:
    """``x @ H`` blockwise over the last dimension."""
    shape = x.shape
    features = shape[-1]
    if features % group_size != 0:
        raise ValueError(f"features {features} not divisible by ConvRot group {group_size}")
    grouped = x.reshape(-1, features // group_size, group_size)
    return grouped.matmul(h.to(dtype = x.dtype, device = x.device)).reshape(shape)


def rotate_convrot_weight_(module: Any, group_size: int) -> None:
    """``W <- W @ blockdiag(H).T`` in place, accumulated in float32 and cast back.

    float32 regardless of the stored dtype: each output element becomes a ``group_size``-term dot
    product, and accumulating that in bfloat16 would spend a visible part of the error budget the
    rotation exists to save. The offline half only runs once, so the upcast is free."""
    import torch

    weight = module.weight.data
    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ValueError(
            f"in_features {in_features} is not divisible by the ConvRot group {group_size}"
        )
    h = build_convrot_hadamard(group_size, device = weight.device, dtype = torch.float32)
    rotated = torch.matmul(
        weight.float().reshape(out_features, in_features // group_size, group_size), h.T
    ).reshape(out_features, in_features)
    module.weight.data = rotated.to(weight.dtype)


@lru_cache(maxsize = None)
def convrot_linear_class() -> Any:
    """The ``nn.Linear`` subclass that rotates its input, built lazily so importing this module
    never imports torch, and built exactly ONCE.

    The cache is not a micro-optimisation, it is the difference between one compiled graph and
    dozens. A class defined inside a function is a NEW class object on every call, and
    ``torch.compile`` guards each frame on ``___check_type_id`` of the modules it closes over. So
    handing every rotated projection its own ConvRotLinear made each one look like a different
    type and retraced the block it lives in: measured 23 recompiles against bfloat16's 1 on the
    same job, 178 s of first-call compile against 14 s, on a denoiser with 350 rotated
    projections. Sharing one class collapses that back to a single trace.

    A CLASS SWAP rather than a wrapper module, and that is load-bearing twice over: the module
    stays an ``nn.Linear``, so torchao's filter and ``quantize_`` treat it exactly as they treat
    an unrotated one, and the state dict keys are unchanged, so the hosted checkpoint still loads
    under ``strict = True`` with no rename. The rotation carries no parameters of its own, so
    there is nothing to serialize."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    class ConvRotLinear(nn.Linear):
        """``nn.Linear`` whose input is block-Hadamard rotated before the matmul."""

        # Set per instance by ``_install_rotation``; the class default exists only so a
        # half-constructed instance cannot silently rotate at some other group.
        convrot_groupsize: int = DEFAULT_CONVROT_GROUPSIZE

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            group = self.convrot_groupsize
            h = build_convrot_hadamard(group, device = x.device, dtype = x.dtype)
            return F.linear(rotate_convrot_activation(x, h, group), self.weight, self.bias)

        def extra_repr(self) -> str:  # pragma: no cover - debug aid
            return f"{super().extra_repr()}, convrot(group={self.convrot_groupsize})"

    return ConvRotLinear


def is_rotated_linear(module: Any) -> bool:
    """True when ``module`` already applies the online rotation to its own input."""
    return type(module).__name__ == "ConvRotLinear" and hasattr(module, "convrot_groupsize")


def _install_rotation(module: Any, group_size: int) -> None:
    """Swap ``module`` onto the rotating subclass, in place."""
    module.convrot_groupsize = int(group_size)
    module.__class__ = convrot_linear_class()


# ── the metadata contract ─────────────────────────────────────────────────────────────────────


def declares_rotation(metadata: Any) -> bool:
    """True when ``metadata`` claims its weights were rotated offline.

    Deliberately keyed on the KEY being populated, not on the value being one this module
    understands: an unknown kind has to read as "a rotation is declared" so the validator can
    refuse it, rather than as "no rotation" so the loader runs rotated weights unrotated."""
    return isinstance(metadata, dict) and metadata.get(ROTATION_KEY) not in (None, "")


def rotation_metadata_error(metadata: Any) -> Optional[str]:
    """Why ``metadata``'s declared rotation is unusable, or None when it is well formed.

    Pure and torch-free, so the prequant validator can call it before anything is built. Checks
    the CONTRACT only (kind, group, fqn list shape); whether those fqns exist on this particular
    model is ``apply_activation_rotation``'s question, since answering it needs the model."""
    if not declares_rotation(metadata):
        return None
    kind = metadata.get(ROTATION_KEY)
    if kind != CONVROT_KIND:
        return f"unsupported activation rotation {kind!r} (this build implements {CONVROT_KIND!r})"
    group = metadata.get(ROTATION_GROUP_KEY)
    if not is_power_of_four(group):
        return (
            f"activation rotation group {group!r} is not a power of 4; the Hadamard is built as a "
            "kron power of H4"
        )
    fqns = metadata.get(ROTATION_FQNS_KEY)
    if not isinstance(fqns, (list, tuple)) or not fqns:
        # An empty list is refused rather than read as "rotate nothing": a builder that failed to
        # record its set would otherwise emit an artifact that loads clean and renders garbage.
        return f"activation rotation records no fqns ({ROTATION_FQNS_KEY} is {fqns!r})"
    if not all(isinstance(fqn, str) and fqn for fqn in fqns):
        return f"activation rotation {ROTATION_FQNS_KEY} has non-string entries"
    if len(set(fqns)) != len(fqns):
        return f"activation rotation {ROTATION_FQNS_KEY} has duplicates"
    return None


def rotation_metadata(group_size: int, fqns: Iterable[str]) -> dict:
    """The metadata fragment an offline builder merges in after rotating ``fqns``.

    Sorted, so two builds of the same model produce identical metadata and a rebuilt artifact can
    be diffed against the shipped one."""
    return {
        ROTATION_KEY: CONVROT_KIND,
        ROTATION_GROUP_KEY: int(group_size),
        ROTATION_FQNS_KEY: sorted(fqns),
    }


# ── the two halves ────────────────────────────────────────────────────────────────────────────


def rotatable_fqns(
    transformer: Any,
    filter_fn: Any,
    group_size: int = DEFAULT_CONVROT_GROUPSIZE,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(rotatable, not_divisible)`` among the Linears ``filter_fn`` selects for quantization.

    The second tuple is why the shipped set is recorded rather than derived: the Hadamard only
    tiles an input axis the group divides, so some quantized Linears are always left plain, and
    which ones is a property of the architecture rather than of a rule worth re-evaluating at
    load time."""
    from torch import nn

    rotatable: list[str] = []
    not_divisible: list[str] = []
    for fqn, module in transformer.named_modules():
        if not isinstance(module, nn.Linear) or not filter_fn(module, fqn):
            continue
        (rotatable if module.in_features % group_size == 0 else not_divisible).append(fqn)
    return tuple(rotatable), tuple(not_divisible)


def rotate_linears_(
    transformer: Any,
    fqns: Iterable[str],
    group_size: int = DEFAULT_CONVROT_GROUPSIZE,
) -> tuple[str, ...]:
    """OFFLINE half: rotate the weights of ``fqns`` and install the online rotation on each.

    Call BEFORE ``quantize_``, on a dense model: the whole point is that the quantizer sees the
    flatter distribution. Returns the fqns rotated, in the order given. Raises on anything it
    cannot rotate, so a builder can never record a set larger than the one it actually applied."""
    from torch import nn

    if not is_power_of_four(group_size):
        raise ValueError(f"ConvRot group size must be a power of 4, got {group_size!r}")
    modules = dict(transformer.named_modules())
    rotated: list[str] = []
    for fqn in fqns:
        module = modules.get(fqn)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"cannot rotate {fqn!r}: not an nn.Linear on this model")
        rotate_convrot_weight_(module, group_size)
        _install_rotation(module, group_size)
        rotated.append(fqn)
    return tuple(rotated)


def apply_activation_rotation(
    transformer: Any,
    metadata: Any,
    *,
    logger: Any = None,
) -> tuple[str, ...]:
    """ONLINE half: install the input rotation on exactly the fqns ``metadata`` records.

    Returns the fqns rotated, or ``()`` when ``metadata`` declares no rotation -- the plain
    artifacts, which have to be left exactly as they are. RAISES on any other outcome: an
    unusable contract, an fqn this model does not have, a target that is not a Linear, an
    ``in_features`` the recorded group does not divide, or a Linear already rotated. The prequant
    loader turns a raise into a refused checkpoint and a dense fallback.

    Call AFTER ``load_state_dict`` and BEFORE ``apply_small_m_padding``: after, because the meta
    retry path rebuilds the module from the config and would discard an earlier swap; before,
    because padding reparents the Linears under a wrapper while the recorded fqns name the
    unwrapped tree."""
    if not declares_rotation(metadata):
        return ()
    problem = rotation_metadata_error(metadata)
    if problem:
        raise ValueError(problem)

    from torch import nn

    group_size = int(metadata[ROTATION_GROUP_KEY])
    fqns = list(metadata[ROTATION_FQNS_KEY])
    modules = dict(transformer.named_modules())
    missing = [fqn for fqn in fqns if fqn not in modules]
    if missing:
        raise ValueError(
            f"activation rotation names {len(missing)} fqn(s) this model does not have "
            f"(e.g. {missing[0]!r}); the checkpoint and this build disagree about the model"
        )
    for fqn in fqns:
        module = modules[fqn]
        if not isinstance(module, nn.Linear):
            raise ValueError(f"activation rotation target {fqn!r} is not an nn.Linear")
        if is_rotated_linear(module):
            raise ValueError(f"activation rotation target {fqn!r} is already rotated")
        if module.in_features % group_size:
            raise ValueError(
                f"activation rotation target {fqn!r} has in_features {module.in_features}, "
                f"which the recorded group {group_size} does not divide"
            )
    # Every target is validated before ANY is swapped. A partial install is the one outcome worse
    # than either end state: the rotated half still renders, just wrongly, so there is nothing to
    # notice and nothing to fall back from.
    for fqn in fqns:
        _install_rotation(modules[fqn], group_size)
    try:
        setattr(
            transformer,
            CONVROT_ATTR,
            {"kind": CONVROT_KIND, "group": group_size, "linears": len(fqns)},
        )
    except Exception:  # noqa: BLE001 -- the marker is a diagnostic, never the mechanism
        pass
    if logger is not None:
        logger.info(
            "diffusion.convrot: installed %s on %d linears at group %d",
            CONVROT_KIND,
            len(fqns),
            group_size,
        )
    return tuple(fqns)
