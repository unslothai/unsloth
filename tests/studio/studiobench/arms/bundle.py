# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bundle-variant arms, and the refusal that keeps them from being skipped quietly.

Three hypotheses cannot be tested by injecting script into a shipped build, because they are
about what React does internally rather than about what the DOM does. They need a dist that was
COMPILED differently, built once in CI and shipped as an ARMPACK next to the benchmark.

  FROZEN ELEMENTS      The retained prefix is re-emitted as the same element objects on every
                       render. If cost is driven by React allocating and cloning per child, this
                       removes the allocation while keeping the fibre count identical.
  PREFIX FOLD          The retained prefix is collapsed behind one memo boundary, so an update
                       anywhere below reaches one child instead of hundreds of siblings. This is
                       the closest thing to the proposed fix, measured before it is written.
  FIBRE-FREE TWIN      The retained prefix is re-emitted as IDENTICAL MARKUP through
                       `dangerouslySetInnerHTML`, so the DOM is unchanged and the fibres are gone.

THE FIBRE-FREE TWIN IS NOT OPTIONAL AND IT IS NOT ONE ARM AMONG THREE. In the shipping build the
number of fibres and the number of DOM nodes move together: every message is both. Every arm that
removes messages removes both at once, so "cost proportional to fibres" and "cost proportional to
DOM nodes" predict the identical result on every one of them. They are UNIDENTIFIABLE. The twin
is the only arm in the whole design that moves one without the other: same nodes, same bytes, same
pixels, no fibres. Without it, no amount of ablation can tell the two hypotheses apart, and a
report that names one of them is guessing.

WHY THE HARD REFUSAL. An armpack that does not match the install cannot be used: the arms are
compiled against a specific dist, and running them against a different one measures the version
difference. The tempting behaviour is to skip those arms and print the rest, which produces a
report that looks complete and silently omits the only arm that could have distinguished the two
live hypotheses. So this module prints ABLATION ARMS NOT AVAILABLE FOR THIS BUILD and exits that
plane of the experiment. The runtime-knob plane still runs; it is the bundle plane that stops.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .manifest import Arm, DeclaredDiff, Invariance, PotencyCounter

BANNER = "ABLATION ARMS NOT AVAILABLE FOR THIS BUILD"


class ArmpackUnavailable(RuntimeError):
    """Raised to end the bundle-arm plane. Carries the banner as its message."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"{BANNER}: {reason}")
        self.reason = reason


ARM_FROZEN_ELEMENTS = Arm(
    arm_id = "BUNDLE_FROZEN_ELEMENTS",
    title = "frozen elements for the retained prefix",
    mechanism = (
        "the prefix is re-emitted as the same element objects every render, so React allocates "
        "nothing per child while the fibre count and the DOM stay identical"
    ),
    invariance = Invariance.EXACT,
    potency = PotencyCounter(
        name = "frozen_element_reuses",
        min_delta = 1,
        direction = "increase",
        description = "element identity was reused for a prefix child at least once",
    ),
    implies_fix = (
        "if this removes the slope, the cost is per-child ALLOCATION rather than the walk itself, "
        "and the fix is to stop producing new element objects for unchanged prefix children"
    ),
    kind = "bundle",
    requires_armpack = True,
)

ARM_PREFIX_FOLD = Arm(
    arm_id = "BUNDLE_PREFIX_FOLD",
    title = "fold the retained prefix behind one memo boundary",
    mechanism = (
        "the prefix becomes a single memoised child, so an update below it reaches one sibling "
        "instead of every retained message"
    ),
    invariance = Invariance.EQUIVALENT,
    declared_diff = DeclaredDiff(
        normaliser = "strip_fold_wrapper",
        keys = ("data-sb-fold-wrapper",),
        rationale = (
            "the fold introduces exactly one wrapper element carrying one marker attribute. The "
            "normaliser removes that element from the canonical serialisation and nothing else; "
            "if anything other than that attribute differs, the arm is voided, because the extra "
            "difference is the one nobody reviewed"
        ),
    ),
    potency = PotencyCounter(
        name = "prefix_fold_children",
        min_delta = 1,
        direction = "increase",
        description = "the number of prefix children collapsed behind the boundary",
    ),
    implies_fix = (
        "if this removes the slope, the fix is structural: give the retained prefix its own memo "
        "boundary, or virtualise the list so the prefix is not a sibling sequence at all"
    ),
    kind = "bundle",
    requires_armpack = True,
)

ARM_FIBRE_FREE_TWIN = Arm(
    arm_id = "BUNDLE_FIBRE_FREE_TWIN",
    title = "fibre-free twin: identical markup, no fibres",
    mechanism = (
        "the retained prefix is re-emitted as identical markup via dangerouslySetInnerHTML. The "
        "DOM nodes, bytes and pixels are the same; the fibres behind them are gone"
    ),
    invariance = Invariance.EXACT,
    potency = PotencyCounter(
        name = "twin_prefix_fibres",
        min_delta = 1,
        direction = "decrease",
        description = (
            "fibres behind the prefix must FALL while the DOM node count stays put; that "
            "combination is the whole point of the arm"
        ),
    ),
    implies_fix = (
        "this arm does not imply a fix, it decides which of two fixes to look for. If the cost "
        "goes with the fibres it is a React structure problem; if it stays with the DOM it is a "
        "style, layout and paint problem, and no amount of memoisation will touch it"
    ),
    kind = "bundle",
    requires_armpack = True,
    notes = (
        "the only arm that breaks fibre/DOM collinearity. Without it, cost proportional to fibres "
        "and cost proportional to DOM are unidentifiable and the report cannot name either"
    ),
)

BUNDLE_ARMS: tuple[Arm, ...] = (
    ARM_FROZEN_ELEMENTS,
    ARM_PREFIX_FOLD,
    ARM_FIBRE_FREE_TWIN,
)


@dataclass(frozen = True)
class ArmpackManifest:
    """What an armpack claims about itself. Read from `armpack.json` inside the pack."""

    armpack_version: str
    built_from_sha: str
    target_dist_digest: str
    arms: Mapping[str, str]  # arm_id -> relative dist directory inside the pack
    root: Path

    @classmethod
    def load(cls, path: str | Path) -> "ArmpackManifest":
        manifest_path = Path(path)
        blob = json.loads(manifest_path.read_text(encoding = "utf-8"))
        missing = [
            key
            for key in ("armpack_version", "built_from_sha", "target_dist_digest", "arms")
            if key not in blob
        ]
        if missing:
            raise ArmpackUnavailable(
                f"{manifest_path} is missing required keys {missing}. An armpack that cannot say "
                "which dist it was built against cannot be matched to one"
            )
        return cls(
            armpack_version = str(blob["armpack_version"]),
            built_from_sha = str(blob["built_from_sha"]),
            target_dist_digest = str(blob["target_dist_digest"]),
            arms = dict(blob["arms"]),
            root = manifest_path.parent,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "armpack_version": self.armpack_version,
            "built_from_sha": self.built_from_sha,
            "target_dist_digest": self.target_dist_digest,
            "arms": dict(self.arms),
            "root": str(self.root),
        }


@dataclass
class ArmpackResolution:
    """The outcome of looking for an armpack that matches this install."""

    available: bool
    reason: str
    manifest: ArmpackManifest | None = None
    arms: list[Arm] = field(default_factory = list)
    searched: list[str] = field(default_factory = list)

    def require(self) -> ArmpackManifest:
        if not self.available or self.manifest is None:
            raise ArmpackUnavailable(self.reason)
        return self.manifest

    def render(self) -> str:
        if self.available and self.manifest is not None:
            return (
                f"armpack {self.manifest.armpack_version} matched "
                f"(dist {self.manifest.target_dist_digest[:12]}, built from "
                f"{self.manifest.built_from_sha[:12]}); bundle arms: "
                + ", ".join(sorted(self.manifest.arms))
            )
        lines = [BANNER, f"  {self.reason}"]
        if self.searched:
            lines.append("  searched:")
            lines.extend(f"    {path}" for path in self.searched)
        lines.append(
            "  The bundle-arm plane of this experiment does not run. In particular the "
            "fibre-free twin does not run, so this report cannot distinguish cost proportional "
            "to fibres from cost proportional to DOM. The runtime-knob plane is unaffected."
        )
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return {
            "available": bool(self.available),
            "reason": self.reason,
            "banner": None if self.available else BANNER,
            "manifest": self.manifest.to_json() if self.manifest else None,
            "arms": [arm.arm_id for arm in self.arms],
            "searched": list(self.searched),
        }


def discover_armpack(
    search_paths: Iterable[str | Path],
    install_dist_digest: str,
    *,
    required_arms: Sequence[Arm] = BUNDLE_ARMS,
) -> ArmpackResolution:
    """Find an armpack whose target digest matches this install, or refuse.

    Matching is on the DIST DIGEST, not on a version string. Two installs claiming the same
    Unsloth version can ship different dists (a local build, a patched install, a different
    Node version producing a different chunk split), and an armpack built against one of them
    measures the build difference when run against the other.
    """

    searched: list[str] = []
    candidates: list[ArmpackManifest] = []
    for base in search_paths:
        base_path = Path(base)
        searched.append(str(base_path))
        manifest_path = (
            base_path if base_path.name == "armpack.json" else base_path / "armpack.json"
        )
        if not manifest_path.is_file():
            continue
        try:
            candidates.append(ArmpackManifest.load(manifest_path))
        except ArmpackUnavailable:
            continue
        except (OSError, json.JSONDecodeError):
            continue

    if not candidates:
        return ArmpackResolution(
            available = False,
            reason = "no armpack.json was found on any search path",
            searched = searched,
        )

    matched = [m for m in candidates if m.target_dist_digest == install_dist_digest]
    if not matched:
        found = ", ".join(sorted({m.target_dist_digest[:12] for m in candidates}))
        return ArmpackResolution(
            available = False,
            reason = (
                f"no armpack targets this install's dist digest {install_dist_digest[:12]}; "
                f"found packs for {found}. Running a mismatched pack would measure the build "
                "difference and report it as a mechanism"
            ),
            searched = searched,
        )

    manifest = matched[0]
    missing = [arm.arm_id for arm in required_arms if arm.arm_id not in manifest.arms]
    if missing:
        return ArmpackResolution(
            available = False,
            reason = (
                f"the matching armpack is missing required arms {missing}. A partial pack is "
                "refused rather than run: the arm most likely to be missing is the fibre-free "
                "twin, and without it the remaining arms cannot identify what they appear to"
            ),
            manifest = manifest,
            searched = searched,
        )

    absent_dirs = [
        arm_id for arm_id, rel in manifest.arms.items() if not (manifest.root / rel).is_dir()
    ]
    if absent_dirs:
        return ArmpackResolution(
            available = False,
            reason = (
                f"the armpack manifest lists arms whose dist directories are absent: "
                f"{sorted(absent_dirs)}"
            ),
            manifest = manifest,
            searched = searched,
        )

    return ArmpackResolution(
        available = True,
        reason = "armpack matched this install",
        manifest = manifest,
        arms = list(required_arms),
        searched = searched,
    )
