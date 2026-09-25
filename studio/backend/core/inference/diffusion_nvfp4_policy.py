# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-layer NVFP4 policies for the image DiTs: a named set of layers at 4 bits over an fp8 model.

A policy puts a named set of layers at 4 bits over fp8: a memory lever, not a speed win, so nvfp4
sits BELOW fp8 in the image auto order. FAILS CLOSED: exact suffixes and asserted counts per rule.
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
    """The model this policy was applied to is not the one it was written for."""


@dataclass(frozen = True)
class Rule:
    """``suffix`` -> ``precision``; ``expect`` is an asserted count, ``prefix`` narrows to one subtree."""

    suffix: str
    precision: str
    expect: int
    prefix: str = ""

    def matches(self, fqn: str) -> bool:
        """Exact dotted-suffix match: ``norm.linear`` does not select ``norm1.linear``."""
        if self.prefix and not fqn.startswith(self.prefix):
            return False
        return fqn == self.suffix or fqn.endswith("." + self.suffix)


@dataclass(frozen = True)
class Admit:
    """A linear the shared filter rejects that this policy quantises anyway. ``shape`` is asserted
    exactly: dropping the floor instead would admit ``t_embedder.mlp.*``, which must stay bf16."""

    suffix: str
    shape: tuple
    expect: int

    def matches(self, fqn: str) -> bool:
        return fqn == self.suffix or fqn.endswith("." + self.suffix)


@dataclass(frozen = True)
class NVFP4Policy:
    """One measured per-layer precision assignment, keyed on lowercased ``base_repos`` (never a sibling's)."""

    policy_id: str
    version: int
    family: str
    base_repos: tuple
    default: str = PRECISION_FP8
    rules: tuple = ()
    admit: tuple = ()
    expected_counts: Mapping = field(default_factory = dict)


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


def _per_block(suffix: str, stem: str, blocks: tuple) -> tuple:
    """One exact rule per listed block: the storage/quality solve ranks layers per block, not per class."""
    return tuple(
        Rule(suffix = suffix, precision = PRECISION_NVFP4, expect = 1, prefix = f"{stem}.{block}.")
        for block in blocks
    )


# Points on the storage/quality frontier whose paired LPIPS(vgg) gap to the model's own fp8 stays
# under 0.05 at the upper 95% bound on held-out pairs. They replace zimg_f8mod_toq34_v1 and
# flux_mod_single_v1, so an artifact built at either is refused on load.
ZIMG_RG76 = NVFP4Policy(
    policy_id = "zimg_rg76_v1",
    version = 1,
    family = "z-image",
    base_repos = ("tongyi-mai/z-image-turbo",),
    rules = (
        Rule(suffix = "attention.to_q", precision = PRECISION_NVFP4, expect = 34),
        Rule(suffix = "attention.to_k", precision = PRECISION_NVFP4, expect = 34),
        *_per_block("feed_forward.w1", "layers", (0, 1, 2, 3, 4, 5)),
        Rule(suffix = "feed_forward.w1", precision = PRECISION_NVFP4, expect = 2, prefix = "noise_refiner."),
    ),
    admit = (Admit(suffix = "adaLN_modulation.0", shape = (256, 15360), expect = 32),),
    # The two ``t_embedder.mlp`` layers cannot be swapped at all and stay dense.
    expected_counts = {PRECISION_NVFP4: 76, PRECISION_FP8: 195, PRECISION_BF16: 5},
)

# One frontier point back from R600, whose paired gap to fp8 measured +0.044 [0.034, 0.053] on 48
# held-out pairs. R600_lr32 sits between the two but needs a low-rank add path the NVFP4 linear lacks.
FLUX_R420 = NVFP4Policy(
    policy_id = "flux_r420_v1",
    version = 1,
    family = "flux.1",
    # schnell only: dev and Krea-dev are separate weights and need their own gate runs.
    base_repos = ("black-forest-labs/flux.1-schnell",),
    rules = (
        Rule(suffix = "norm_out.linear", precision = PRECISION_NVFP4, expect = 1),
        *_per_block("attn.to_k", "single_transformer_blocks", tuple(range(15, 38))),
        *_per_block("attn.to_q", "single_transformer_blocks", tuple(range(15, 38))),
        *_per_block(
            "attn.to_v", "single_transformer_blocks", (19, 21, 24, 26) + tuple(range(28, 38))
        ),
        *_per_block("norm.linear", "single_transformer_blocks", tuple(range(11, 38))),
        *_per_block("proj_mlp", "single_transformer_blocks", tuple(range(16, 38))),
        *_per_block("proj_out", "single_transformer_blocks", tuple(range(18, 38))),
        *_per_block("attn.add_k_proj", "transformer_blocks", (3, 4, 6, 15)),
        *_per_block("attn.add_q_proj", "transformer_blocks", (0, 4, 7, 9)),
        *_per_block("attn.to_add_out", "transformer_blocks", (2,)),
        *_per_block("attn.to_k", "transformer_blocks", (0, 4, 9)),
        *_per_block("attn.to_q", "transformer_blocks", (0, 1, 5, 6)),
        *_per_block("attn.to_v", "transformer_blocks", (6,)),
        *_per_block("ff.net.0.proj", "transformer_blocks", (0, 1, 3, 4, 5)),
        *_per_block("ff.net.2", "transformer_blocks", (0,) + tuple(range(2, 8))),
        *_per_block("ff_context.net.0.proj", "transformer_blocks", (12, 13, 14)),
        *_per_block(
            "ff_context.net.2",
            "transformer_blocks",
            (0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 16, 18),
        ),
        *_per_block("norm1.linear", "transformer_blocks", (5,)),
        *_per_block(
            "norm1_context.linear",
            "transformer_blocks",
            (3, 4, 7, 8, 9, 12, 13, 14, 15, 16, 18),
        ),
    ),
    expected_counts = {PRECISION_NVFP4: 187, PRECISION_FP8: 312, PRECISION_BF16: 3},
)

# One frontier point back from R040, whose paired gap to fp8 measured +0.039 [0.022, 0.057] on 48
# held-out pairs: the upper bound missed 0.05 there.
QWEN21_R020 = NVFP4Policy(
    policy_id = "qwen21_r020_v1",
    version = 1,
    family = "qwen-image-2.1",
    base_repos = ("qwen/qwen-image-2.1",),
    rules = (
        Rule(suffix = "norm_out.linear", precision = PRECISION_NVFP4, expect = 1),
        *_per_block("attn.to_k", "transformer_blocks", (5, 7, 13) + tuple(range(15, 28))),
        *_per_block("attn.to_q", "transformer_blocks", (1, 4, 18) + tuple(range(20, 28)) + (29,)),
        *_per_block(
            "img_mlp.gate_layer", "transformer_blocks", (8, 13, 17, 18, 19) + tuple(range(21, 29))
        ),
        *_per_block("img_mlp.proj", "transformer_blocks", (0, 1, 12, 13, 17, 18)),
    ),
    expected_counts = {PRECISION_NVFP4: 48, PRECISION_FP8: 181, PRECISION_BF16: 3},
)

# 2512 is its own weights under the qwen-image family: qwen_p02_v1 stays the Qwen/Qwen-Image policy.
QWEN2512_M120_ATTN8 = NVFP4Policy(
    policy_id = "qwen2512_m120_attn8_v1",
    version = 1,
    family = "qwen-image",
    base_repos = ("qwen/qwen-image-2512",),
    rules = tuple(
        Rule(suffix = suffix, precision = PRECISION_NVFP4, expect = 60)
        for suffix in (
            "attn.to_k",
            "img_mod.1",
            "txt_mod.1",
            "attn.to_add_out",
            "attn.add_q_proj",
            "attn.to_v",
            "attn.add_k_proj",
            "attn.to_q",
            "attn.to_out.0",
            "attn.add_v_proj",
        )
    ),
    expected_counts = {PRECISION_NVFP4: 600, PRECISION_FP8: 243, PRECISION_BF16: 3},
)

NVFP4_POLICIES: tuple = (ZIMG_RG76, FLUX_R420, QWEN21_R020, QWEN2512_M120_ATTN8, QWEN_P02)


def policy_by_id(policy_id: Any) -> Optional[NVFP4Policy]:
    wanted = str(policy_id or "").strip()
    for policy in NVFP4_POLICIES:
        if policy.policy_id == wanted:
            return policy
    return None


def policy_expected_counts(policy: NVFP4Policy) -> dict:
    return {
        str(key): int(value) for key, value in dict(policy.expected_counts).items() if int(value)
    }


def resolve_policy(family: Any, base_repo: Any = None) -> Optional[NVFP4Policy]:
    """The policy for ``(family, base_repo)``, or None; an unnamed base is None even if the family has one."""
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
    """``{fqn: precision}`` for EVERY Linear, or raise ``PolicyMismatch``; rules apply in order, first claim wins."""
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
    """True when ``metadata`` has the policy KEY; an unreadable block must refuse, not read as "no policy"."""
    return isinstance(metadata, dict) and metadata.get(NVFP4_POLICY_KEY) not in (None, "")


def policy_metadata_error(metadata: Any) -> Optional[str]:
    """Why ``metadata``'s declared policy breaks the contract, or None. Torch-free."""
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
    """Raise unless every assigned layer holds its precision's weight class (torchao can skip silently)."""
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
    """Apply ``policy`` in place via two disjoint ``quantize_`` passes, NVFP4 first; discard on a raise."""
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
