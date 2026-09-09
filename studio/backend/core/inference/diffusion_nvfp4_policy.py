# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-layer NVFP4 policies for the image DiTs.

Whole-model NVFP4 pays on the video denoisers (smaller AND faster than fp8). On the image ones it
does not: the campaign measured every admitted linear at NVFP4 as a quality loss the render shows,
and the arms that held up quantise only a small, named set of layers to 4 bits and leave the rest
at fp8. Those sets are what this module holds:

  * z-image  ``F8mod_toq34``  the 34 ``attention.to_q`` projections at NVFP4, everything else fp8,
    and the 32 ``adaLN_modulation.0`` modulation projections pulled INTO the fp8 set even though
    their 256-wide input is below the min_features floor (they are the family's second-largest
    weight block, and fp8 on them is where most of its saving comes from);
  * flux.1   ``mod_single``    the 38 single-block ``norm.linear`` modulation projections;
  * qwen-image ``P02``         the 120 ``img_mod.1`` / ``txt_mod.1`` modulation projections.

These are memory levers at fp8-parity quality, not speed wins: the NVFP4 layers are ``to_q`` and
M=1 modulation projections far below the GEMM crossover, which is why nvfp4 sits BELOW fp8 in the
image auto order.

Everything here FAILS CLOSED. A policy names its layers by exact dotted suffix and asserts a count
for every rule, for the admitted set and for the final assignment, so a diffusers rename or a
config change that moves one linear raises at build time instead of shipping an artifact whose
per-layer precisions are not the ones anything measured. A checkpoint carries the same counts and
the loader re-resolves the in-tree policy against them, so the two halves cannot drift apart
either. The alternative -- deriving the set from a rule at load time -- is how a policy silently
becomes a different policy.

Matching is by dotted SUFFIX, never substring: ``norm.linear`` must not select flux's
``transformer_blocks.N.norm1.linear`` or its top-level ``norm_out.linear``, which are different
layers of a different width that no gate ran on.

torch is imported inside the functions, matching the other lazily-loaded inference helpers, so
reading the tables or the metadata contract costs no import.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

# The precision tokens an assignment uses. The two quantised ones are the scheme tokens
# ``diffusion_transformer_quant`` already defines (spelled out here so reading the tables costs no
# import); "bf16" is "left dense", which is not a scheme and has no token there.
PRECISION_NVFP4 = "nvfp4"
PRECISION_FP8 = "fp8"
PRECISION_BF16 = "bf16"

# ── the metadata contract, carried in the prequant checkpoint's own ``metadata`` dict ──────────
# The key a checkpoint's policy block lives under, and the KIND that block declares. The kind is
# versioned separately from the individual policies: it describes the SHAPE of the block, so a
# build that adds a field bumps it, while a retuned z-image set bumps ``zimg_..._v2`` instead.
NVFP4_POLICY_KEY = "nvfp4_policy"
NVFP4_POLICY_KIND = "unsloth_nvfp4_layer_policy_v1"


class PolicyMismatch(ValueError):
    """The model this policy was applied to is not the model it was written for.

    Raised rather than warned: every count here was measured on one specific base repo at one
    diffusers version, so a rule that now selects a different number of layers is a DIFFERENT
    model, and building it anyway produces an artifact whose quality nothing has measured."""


@dataclass(frozen = True)
class Rule:
    """``suffix`` -> ``precision``, for the layers of an admitted set that match it.

    ``expect`` is the number of layers this rule must select, asserted at assignment time.
    ``prefix`` narrows the rule to one subtree (flux's single blocks carry a ``norm.linear`` its
    double blocks spell ``norm1.linear``, and only the single ones were gated)."""

    suffix: str
    precision: str
    expect: int
    prefix: str = ""

    def matches(self, fqn: str) -> bool:
        """Exact dotted-suffix match, never a substring one.

        ``to_q`` selects ``layers.0.attention.to_q`` and not ``...attention.to_q_extra``;
        ``norm.linear`` selects neither ``norm1.linear`` nor ``norm_out.linear``."""
        if self.prefix and not fqn.startswith(self.prefix):
            return False
        return fqn == self.suffix or fqn.endswith("." + self.suffix)


@dataclass(frozen = True)
class Admit:
    """A linear the shared filter rejects that this policy quantises anyway.

    z-image's ``adaLN_modulation.0`` is (256, 15360): its INPUT is below the 512 floor, but it is
    the second-largest weight block in the model and leaving 32 of them dense gives up most of the
    saving. ``shape`` is ``(in_features, out_features)`` and is asserted exactly -- the floor is
    being overridden for a specific measured layer, so a layer of another width wearing the same
    name is not it. Dropping the floor to 256 instead is what this exists to avoid: that would
    also admit ``t_embedder.mlp.*``, which cannot be quantised at all (``TimestepEmbedder.forward``
    reads ``mlp[0].weight.dtype``)."""

    suffix: str
    shape: tuple
    expect: int

    def matches(self, fqn: str) -> bool:
        return fqn == self.suffix or fqn.endswith("." + self.suffix)


@dataclass(frozen = True)
class NVFP4Policy:
    """One measured per-layer precision assignment, for one family on one set of base repos.

    ``base_repos`` are lowercased upstream ids: a policy is a claim about weights, and a sibling
    checkpoint (flux dev beside schnell) is a different set of weights whose gate has not run.
    ``expected_counts`` is the whole assignment, bf16 included, so a linear that stops being a
    linear is caught as loudly as one that changes rule."""

    policy_id: str
    version: int
    family: str
    base_repos: tuple
    default: str = PRECISION_FP8
    rules: tuple = ()
    admit: tuple = ()
    expected_counts: Mapping = field(default_factory = dict)


# ── the tables ────────────────────────────────────────────────────────────────────────────────
# Every count below was verified against the diffusers module tree instantiated on the meta device
# for the base repo named in the row, and matches the linear census the campaign measured on
# (z-image 239 admitted, flux 499, qwen-image 843).

ZIMAGE_F8MOD_TOQ34 = NVFP4Policy(
    policy_id = "zimg_f8mod_toq34_v1",
    version = 1,
    family = "z-image",
    base_repos = ("tongyi-mai/z-image-turbo",),
    rules = (Rule(suffix = "attention.to_q", precision = PRECISION_NVFP4, expect = 34),),
    admit = (Admit(suffix = "adaLN_modulation.0", shape = (256, 15360), expect = 32),),
    # 276 linears: 34 to_q at NVFP4, 237 at fp8 (205 admitted by the filter plus the 32 admits),
    # and 5 dense -- the two ``t_embedder.mlp`` layers, which cannot be swapped at all, plus the
    # 64-wide patch embedder and final layer.
    expected_counts = {PRECISION_NVFP4: 34, PRECISION_FP8: 237, PRECISION_BF16: 5},
)

FLUX_MOD_SINGLE = NVFP4Policy(
    policy_id = "flux_mod_single_v1",
    version = 1,
    family = "flux.1",
    # schnell only: dev and Krea-dev are separate weights and get their own gate runs before they
    # get a row here.
    base_repos = ("black-forest-labs/flux.1-schnell",),
    rules = (
        Rule(
            suffix = "norm.linear",
            precision = PRECISION_NVFP4,
            expect = 38,
            prefix = "single_transformer_blocks.",
        ),
    ),
    # 502 linears: 38 single-block modulation projections at NVFP4, 461 at fp8, and the 3 the
    # filter leaves dense (``x_embedder`` and ``proj_out`` are 64-wide, the timestep embedder's
    # first layer is 256-wide).
    expected_counts = {PRECISION_NVFP4: 38, PRECISION_FP8: 461, PRECISION_BF16: 3},
)

QWEN_P02 = NVFP4Policy(
    policy_id = "qwen_p02_v1",
    version = 1,
    family = "qwen-image",
    # qwen-image-edit is a separate row with its own gate.
    base_repos = ("qwen/qwen-image",),
    rules = (
        Rule(suffix = "img_mod.1", precision = PRECISION_NVFP4, expect = 60),
        Rule(suffix = "txt_mod.1", precision = PRECISION_NVFP4, expect = 60),
    ),
    # 846 linears: 120 modulation projections at NVFP4, 723 at fp8, 3 dense (``img_in`` and
    # ``proj_out`` are 64-wide, the timestep embedder's first layer is 256-wide).
    expected_counts = {PRECISION_NVFP4: 120, PRECISION_FP8: 723, PRECISION_BF16: 3},
)

NVFP4_POLICIES: tuple = (ZIMAGE_F8MOD_TOQ34, FLUX_MOD_SINGLE, QWEN_P02)


def policy_by_id(policy_id: Any) -> Optional[NVFP4Policy]:
    """The in-tree policy with this id, or None. Used to re-resolve a checkpoint's declaration."""
    wanted = str(policy_id or "").strip()
    for policy in NVFP4_POLICIES:
        if policy.policy_id == wanted:
            return policy
    return None


def policy_expected_counts(policy: NVFP4Policy) -> dict:
    """``policy.expected_counts`` as a ``Counter`` comparison sees it: zero entries dropped.

    A Counter never records a precision no layer took, so a table that spells out ``bf16: 0``
    would otherwise fail against an assignment that is exactly right."""
    return {
        str(key): int(value) for key, value in dict(policy.expected_counts).items() if int(value)
    }


def resolve_policy(family: Any, base_repo: Any = None) -> Optional[NVFP4Policy]:
    """The policy for ``(family, base_repo)``, or None when there is none.

    A policy is keyed on the BASE, not on the family: the layer sets were solved against one
    checkpoint's weights, and flux dev is not flux schnell. Mirrors are canonicalised first, since
    a local mirror is the same weights under another name.

    An unnamed base answers None even when the family has exactly one policy today. Inheriting a
    verdict is the failure mode this whole module exists to prevent, and "the family has one
    policy" is a fact about the table at this commit, not about the caller's model: the day a
    second base is gated, every anonymous load would silently pick up the first one's precisions.
    Callers that know their base pass it; callers that do not get the whole-model path."""
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
    """``{fqn: precision}`` for EVERY Linear in ``transformer``, or raise ``PolicyMismatch``.

    The admitted set is the shared runtime filter's, so a policy layer is one the whole-model path
    would have quantised too, plus this policy's explicit ``admit`` entries. Rules are applied in
    order and the first one to claim a layer keeps it; whatever the rules do not claim takes the
    policy's default, and every Linear outside the admitted set is dense bf16.

    Four assertions, each of which is a rename or a config change caught at build time rather than
    a differently-quantised model shipped: the admit count and shape, each rule's count, and the
    final ``Counter`` against the policy's expected totals."""
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
    # "lora_" keeps a baked adapter's side path high precision, and require_bf16 skips the layers
    # the fp8 pass would abort on. Same call the runtime and the builder make, so "admitted" means
    # one thing in all three.
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
        # Among the layers no earlier rule claimed, so "first wins" is what the count asserts too.
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
    """The metadata fragment an offline builder merges in after applying ``policy``.

    Sorted, so two builds of the same model produce identical metadata and a rebuilt artifact can
    be diffed against the shipped one. The fqn list is recorded rather than re-derived for the
    same reason the rotation's is: at load time the model is a skeleton on ``meta``, and a rule
    that has drifted would re-derive a different set with nothing saying so."""
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
            # Set by --bake-activation-scales; the flashinfer backend refuses an artifact without
            # baked scales and falls back to torchao.
            "activation_scales_baked": bool(activation_scales_baked),
            "gptq": bool(gptq),
        }
    }


def declares_policy(metadata: Any) -> bool:
    """True when ``metadata`` claims a per-layer policy was applied.

    Keyed on the KEY being populated, not on the block being one this build understands: an
    unreadable block has to read as "a policy is declared" so the validator can refuse it, rather
    than as "no policy" so the loader treats a mixed-precision artifact as a whole-model one."""
    return isinstance(metadata, dict) and metadata.get(NVFP4_POLICY_KEY) not in (None, "")


def policy_metadata_error(metadata: Any) -> Optional[str]:
    """Why ``metadata``'s declared policy is unusable, or None when it is well formed.

    Pure and torch-free, so the prequant validator can call it before anything is built. Checks
    the CONTRACT only (kind, ids, counts, fqn list shape); whether the declaration agrees with the
    policy this build resolves for the artifact's family and base is the validator's question,
    since answering it needs the family tables."""
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
        # An empty list is refused rather than read as "quantise nothing to 4 bits": a builder
        # that failed to record its set would otherwise emit an artifact that loads clean and
        # renders from precisions nobody chose.
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


# What ``quantize_`` leaves on a module's ``weight`` for each precision, by class NAME: torchao's
# subclasses are re-exported under several module paths and the prototype ones move between
# releases, and asking for the name keeps this module torch-lazy. A layer the policy left dense
# keeps a plain ``nn.Parameter``.
_EXPECTED_WEIGHT_CLASS = {
    PRECISION_NVFP4: "NVFP4Tensor",
    PRECISION_FP8: "Float8Tensor",
    PRECISION_BF16: "Parameter",
}


def _verify_weight_types(transformer: Any, assignment: Mapping) -> None:
    """Raise unless every assigned layer ended up holding the weight class its precision implies.

    The two passes are filtered by fqn set, so a torchao that silently skipped a layer (an
    unsupported shape, a config it declined) would leave it dense with nothing in the logs and an
    artifact whose metadata claims a precision it does not have. Checked here rather than trusted:
    the whole point of a policy is that the per-layer precisions are exactly the measured ones."""
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
    """Apply ``policy`` to ``transformer`` in place. Returns the assignment it applied.

    Two ``quantize_`` passes over disjoint fqn sets, NVFP4 first. The order matters: after pass 1
    the NVFP4 layers no longer hold a plain ``nn.Parameter``, so pass 2's filter can require one
    and a layer can never be quantised twice however the sets are computed.

    Both configs come from the shared factory, so they are the ones the runtime path builds (and
    go through ``_quiet_config``, whose default would otherwise change every render). Not
    best-effort: a raise here means the module is partly quantised, which is the caller's cue to
    throw it away, not to save it."""
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
        # Belt and braces on the disjointness: a quantised weight is no longer a plain Parameter
        # (torchao returns the tensor subclass itself), and fp8 asserts a bf16 input weight.
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
