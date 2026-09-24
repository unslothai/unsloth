# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-layer NVFP4 policies for the image DiTs: a named set of layers at 4 bits over an fp8 model.

FAILS CLOSED: layers match by exact dotted SUFFIX and every rule asserts a count, so a diffusers
rename raises at build time instead of shipping precisions nobody measured.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

PRECISION_NVFP4 = "nvfp4"
PRECISION_FP8 = "fp8"
PRECISION_BF16 = "bf16"

NVFP4_POLICY_KEY = "nvfp4_policy"
NVFP4_POLICY_KIND = "unsloth_nvfp4_layer_policy_v1"


class PolicyMismatch(ValueError):
    """The model is not the one this policy was written for. Raised, never warned."""


@dataclass(frozen = True)
class Rule:
    """``expect`` is asserted at assignment time; ``prefix`` narrows the rule to one subtree."""

    suffix: str
    precision: str
    expect: int
    prefix: str = ""

    def matches(self, fqn: str) -> bool:
        """Exact dotted suffix: ``norm.linear`` selects neither ``norm1.linear`` nor ``norm_out``."""
        if self.prefix and not fqn.startswith(self.prefix):
            return False
        return fqn == self.suffix or fqn.endswith("." + self.suffix)


@dataclass(frozen = True)
class Admit:
    """A linear the shared filter rejects that this policy quantises anyway, at an exact ``shape``.
    Dropping the floor instead would admit ``t_embedder.mlp.*``, which must stay bf16
    (``TimestepEmbedder.forward`` reads ``mlp[0].weight.dtype``)."""

    suffix: str
    shape: tuple
    expect: int

    def matches(self, fqn: str) -> bool:
        return fqn == self.suffix or fqn.endswith("." + self.suffix)


@dataclass(frozen = True)
class NVFP4Policy:
    """Keyed on lowercased ``base_repos``: a sibling checkpoint is different, ungated weights."""

    policy_id: str
    version: int
    family: str
    base_repos: tuple
    default: str = PRECISION_FP8
    rules: tuple = ()
    admit: tuple = ()
    expected_counts: Mapping = field(default_factory = dict)


ZIMAGE_F8MOD_TOQ34 = NVFP4Policy(
    policy_id = "zimg_f8mod_toq34_v1",
    version = 1,
    family = "z-image",
    base_repos = ("tongyi-mai/z-image-turbo",),
    rules = (Rule(suffix = "attention.to_q", precision = PRECISION_NVFP4, expect = 34),),
    admit = (Admit(suffix = "adaLN_modulation.0", shape = (256, 15360), expect = 32),),
    # The two ``t_embedder.mlp`` layers cannot be swapped at all and stay dense.
    expected_counts = {PRECISION_NVFP4: 34, PRECISION_FP8: 237, PRECISION_BF16: 5},
)

FLUX_MOD_SINGLE = NVFP4Policy(
    policy_id = "flux_mod_single_v1",
    version = 1,
    family = "flux.1",
    # schnell only: dev and Krea-dev are separate weights and need their own gate runs.
    base_repos = ("black-forest-labs/flux.1-schnell",),
    rules = (
        Rule(
            suffix = "norm.linear",
            precision = PRECISION_NVFP4,
            expect = 38,
            prefix = "single_transformer_blocks.",
        ),
    ),
    expected_counts = {PRECISION_NVFP4: 38, PRECISION_FP8: 461, PRECISION_BF16: 3},
)

QWEN_P02 = NVFP4Policy(
    policy_id = "qwen_p02_v1",
    version = 1,
    family = "qwen-image",
    base_repos = ("qwen/qwen-image",),
    rules = (
        Rule(suffix = "img_mod.1", precision = PRECISION_NVFP4, expect = 60),
        Rule(suffix = "txt_mod.1", precision = PRECISION_NVFP4, expect = 60),
    ),
    expected_counts = {PRECISION_NVFP4: 120, PRECISION_FP8: 723, PRECISION_BF16: 3},
)

NVFP4_POLICIES: tuple = (ZIMAGE_F8MOD_TOQ34, FLUX_MOD_SINGLE, QWEN_P02)


def policy_by_id(policy_id: Any) -> Optional[NVFP4Policy]:
    wanted = str(policy_id or "").strip()
    for policy in NVFP4_POLICIES:
        if policy.policy_id == wanted:
            return policy
    return None


def policy_expected_counts(policy: NVFP4Policy) -> dict:
    """Zero entries dropped: a Counter never records a precision no layer took."""
    return {
        str(key): int(value) for key, value in dict(policy.expected_counts).items() if int(value)
    }


def resolve_policy(family: Any, base_repo: Any = None) -> Optional[NVFP4Policy]:
    """Keyed on the BASE: an unnamed base answers None even where the family has one policy, or a
    second gated base would hand every anonymous load the first one's precisions."""
    fam = str(family or "").strip().lower()
    if not fam:
        return None
    candidates = [policy for policy in NVFP4_POLICIES if policy.family == fam]
    if not candidates or not base_repo:
        return None
    from .diffusion_families import canonical_base

    base = canonical_base(str(base_repo).strip()).strip().lower()
    for policy in candidates:
        if base in policy.base_repos:
            return policy
    return None


def assign_precisions(
    transformer: Any,
    policy: NVFP4Policy,
    *,
    min_features: Optional[int] = None,
    require_divisible: Optional[int] = None,
) -> dict:
    import torch

    from .diffusion_transformer_quant import (
        DEFAULT_MIN_LINEAR_FEATURES,
        TQ_NVFP4,
        divisible_for_scheme,
        make_filter_fn,
    )

    if min_features is None:
        min_features = DEFAULT_MIN_LINEAR_FEATURES
    if require_divisible is None:
        require_divisible = divisible_for_scheme(TQ_NVFP4)
    base_filter = make_filter_fn(
        min_features,
        ("lora_",),
        require_bf16 = True,
        require_divisible = require_divisible,
    )
    linears = [
        (fqn, module)
        for fqn, module in transformer.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    modules = dict(linears)
    admitted = {fqn for fqn, module in linears if base_filter(module, fqn)}
    for admit in policy.admit:
        hits = sorted(fqn for fqn, _ in linears if admit.matches(fqn))
        if len(hits) != admit.expect:
            raise PolicyMismatch(
                f"policy {policy.policy_id!r} admits {admit.suffix!r} expecting "
                f"{admit.expect} layers, found {len(hits)}"
            )
        for fqn in hits:
            module = modules[fqn]
            shape = (int(module.in_features), int(module.out_features))
            if shape != tuple(admit.shape):
                raise PolicyMismatch(
                    f"policy {policy.policy_id!r} admits {fqn!r} at {tuple(admit.shape)}, this "
                    f"model has {shape}"
                )
        admitted.update(hits)
    assignment: dict = {}
    for rule in policy.rules:
        hits = sorted(fqn for fqn in admitted if fqn not in assignment and rule.matches(fqn))
        if len(hits) != rule.expect:
            raise PolicyMismatch(
                f"policy {policy.policy_id!r} rule {rule.prefix}{rule.suffix!r} -> "
                f"{rule.precision} expects {rule.expect} layers, found {len(hits)}"
            )
        for fqn in hits:
            assignment[fqn] = rule.precision
    for fqn in admitted:
        assignment.setdefault(fqn, policy.default)
    for fqn, _ in linears:
        assignment.setdefault(fqn, PRECISION_BF16)
    counts = dict(Counter(assignment.values()))
    expected = policy_expected_counts(policy)
    if counts != expected:
        raise PolicyMismatch(
            f"policy {policy.policy_id!r} assigned {counts} over {len(linears)} linears, expected "
            f"{expected}"
        )
    return assignment


def policy_metadata(
    policy: NVFP4Policy,
    assignment: Mapping,
    *,
    activation_scales_baked: bool = False,
    gptq: bool = False,
) -> dict:
    """Sorted for byte-stable builds; the fqn list is recorded, not re-derived from a drifting rule."""
    counts = Counter(assignment.values())
    return {
        NVFP4_POLICY_KEY: {
            "kind": NVFP4_POLICY_KIND,
            "policy_id": policy.policy_id,
            "policy_version": int(policy.version),
            "counts": {str(key): int(counts[key]) for key in sorted(counts)},
            "nvfp4_fqns": sorted(
                fqn for fqn, precision in assignment.items() if precision == PRECISION_NVFP4
            ),
            # The flashinfer backend falls back to torchao without baked scales.
            "activation_scales_baked": bool(activation_scales_baked),
            "gptq": bool(gptq),
        }
    }


def declares_policy(metadata: Any) -> bool:
    """Keyed on the KEY: an unreadable block must refuse, not read as "no policy"."""
    return isinstance(metadata, dict) and metadata.get(NVFP4_POLICY_KEY) not in (None, "")


def policy_metadata_error(metadata: Any) -> Optional[str]:
    if not declares_policy(metadata):
        return None
    block = metadata.get(NVFP4_POLICY_KEY)
    if not isinstance(block, dict):
        return f"nvfp4 policy block is {type(block).__name__}, not a dict"
    kind = block.get("kind")
    if kind != NVFP4_POLICY_KIND:
        return f"unsupported nvfp4 policy {kind!r} (this build implements {NVFP4_POLICY_KIND!r})"
    policy_id = block.get("policy_id")
    if not isinstance(policy_id, str) or not policy_id.strip():
        return f"nvfp4 policy records no policy_id ({policy_id!r})"
    version = block.get("policy_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        return f"nvfp4 policy {policy_id!r} records policy_version {version!r}"
    counts = block.get("counts")
    if not isinstance(counts, dict) or not counts:
        return f"nvfp4 policy {policy_id!r} records no counts ({counts!r})"
    for key, value in counts.items():
        if not isinstance(key, str) or not isinstance(value, int) or isinstance(value, bool):
            return f"nvfp4 policy {policy_id!r} has a malformed count entry {key!r}: {value!r}"
    fqns = block.get("nvfp4_fqns")
    if not isinstance(fqns, (list, tuple)) or not fqns:
        return f"nvfp4 policy {policy_id!r} records no nvfp4_fqns ({fqns!r})"
    if not all(isinstance(fqn, str) and fqn for fqn in fqns):
        return f"nvfp4 policy {policy_id!r} nvfp4_fqns has non-string entries"
    if len(set(fqns)) != len(fqns):
        return f"nvfp4 policy {policy_id!r} nvfp4_fqns has duplicates"
    declared = counts.get(PRECISION_NVFP4)
    if declared is not None and declared != len(fqns):
        return (
            f"nvfp4 policy {policy_id!r} counts {declared} nvfp4 layers but lists {len(fqns)} "
            "nvfp4_fqns"
        )
    return None


# By class NAME: torchao moves these between module paths across releases.
_EXPECTED_WEIGHT_CLASS = {
    PRECISION_NVFP4: "NVFP4Tensor",
    PRECISION_FP8: "Float8Tensor",
    PRECISION_BF16: "Parameter",
}


def _verify_weight_types(transformer: Any, assignment: Mapping) -> None:
    """Raise unless every layer holds its precision's weight class: torchao skips layers silently."""
    wrong: list = []
    for fqn, module in transformer.named_modules():
        precision = assignment.get(fqn)
        if precision is None:
            continue
        want = _EXPECTED_WEIGHT_CLASS.get(precision)
        if want is None:
            continue
        got = type(getattr(module, "weight", None)).__name__
        if got != want:
            wrong.append(f"{fqn}: {precision} wanted {want}, got {got}")
    if wrong:
        raise PolicyMismatch(
            f"{len(wrong)} layers did not take the precision the policy assigned:\n  "
            + "\n  ".join(wrong[:20])
        )


def quantize_with_policy(
    transformer: Any,
    policy: NVFP4Policy,
    *,
    min_features: Optional[int] = None,
    fast_accum: Optional[bool] = None,
    logger: Any = None,
) -> dict:
    """NVFP4 pass first, then fp8 over plain Parameters only. A raise leaves it partly quantised."""
    import torch
    from torchao.quantization import quantize_

    from .diffusion_transformer_quant import TQ_FP8, TQ_NVFP4, _make_quant_config

    assignment = assign_precisions(transformer, policy, min_features = min_features)
    nvfp4_fqns = {fqn for fqn, precision in assignment.items() if precision == PRECISION_NVFP4}
    fp8_fqns = {fqn for fqn, precision in assignment.items() if precision == PRECISION_FP8}

    def nvfp4_filter(module: Any, fqn: str = "") -> bool:
        return fqn in nvfp4_fqns

    def fp8_filter(module: Any, fqn: str = "") -> bool:
        if fqn not in fp8_fqns:
            return False
        weight = getattr(module, "weight", None)
        return type(weight) is torch.nn.Parameter and weight.dtype == torch.bfloat16

    quantize_(transformer, _make_quant_config(TQ_NVFP4), filter_fn = nvfp4_filter)
    quantize_(
        transformer,
        _make_quant_config(TQ_FP8, fast_accum = fast_accum),
        filter_fn = fp8_filter,
    )
    _verify_weight_types(transformer, assignment)
    if logger is not None:
        counts = Counter(assignment.values())
        logger.info(
            "diffusion.nvfp4_policy: %s v%d applied (%s)",
            policy.policy_id,
            policy.version,
            ", ".join(f"{key} {counts[key]}" for key in sorted(counts)),
        )
    return assignment
