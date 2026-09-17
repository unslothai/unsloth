# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The declared unsloth_zoo floor has to supply the helper vision GRPO refuses to run without.

Both GRPO logprob paths raise when `unsloth_zoo.rl_replacements.grpo_vision_chunks` is missing,
because a local fallback could only cover the no-grad half: the gradient half is inside the zoo's
own `grpo_accumulated_loss`, an older copy of which ignores the keys it does not know and, for a
model with no `image_grid_thw` to slice by, replaces `pixel_values` with `None` outright. Half a
fallback makes the two policies disagree, which is the defect this whole path exists to remove.

So the guard is only honest while the floor in pyproject.toml names a release that carries the
helper. With the floor left at 2026.9.4 (the newest release on PyPI that does not export it) an
otherwise dependency-compliant install raises for every vision batch, including the single image
grid models that worked before. unslothai/unsloth#6960, unslothai/unsloth-zoo#1233.

That floor is deferred for now: 2026.9.5 is not published, so declaring it makes the whole
package unresolvable, which is worse than the raise it prevents. The floor assertion below is
skipped until the release ships, and lifting the skip is the one line that re-arms it. The
message assertion stays live: whatever the metadata says, the raise has to name the release that
fixes it.

Reads files only, so it runs on the Windows and macOS runners too.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version


REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"
RL_REPLACEMENTS = REPO / "unsloth" / "models" / "rl_replacements.py"

# The first unsloth_zoo release that carries grpo_vision_chunks. The same number
# unslothai/unsloth#11137 moves the floor to for the TRL ceiling, deliberately: one floor, not
# two, or whichever is higher silently decides what a user resolves.
ZOO_FLOOR_WITH_THE_VISION_CHUNKER = Version("2026.9.5")


def _zoo_requirements() -> list[Requirement]:
    import tomllib

    data = tomllib.loads(PYPROJECT.read_text(encoding = "utf-8"))
    project = data.get("project") or {}
    raws: list[str] = list(project.get("dependencies") or [])
    for extra in (project.get("optional-dependencies") or {}).values():
        raws.extend(extra)
    out = []
    for raw in raws:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        if req.name.lower().replace("_", "-") == "unsloth-zoo" and req.specifier:
            out.append(req)
    return out


def _the_chunker_is_required() -> bool:
    source = RL_REPLACEMENTS.read_text(encoding = "utf-8")
    return "grpo_vision_chunks" in source and "needs an unsloth_zoo build that exports" in source


@pytest.mark.skip(
    reason = f"unsloth_zoo {ZOO_FLOOR_WITH_THE_VISION_CHUNKER} is not on PyPI yet, so the floor "
    "is deferred rather than declared. Drop this skip with the release."
)
def test_the_declared_zoo_floor_carries_the_chunker_the_vision_paths_require() -> None:
    if not _the_chunker_is_required():
        pytest.skip("rl_replacements.py no longer requires grpo_vision_chunks outright")
    reqs = _zoo_requirements()
    assert reqs, (
        "vision GRPO raises without unsloth_zoo.grpo_vision_chunks and pyproject.toml names "
        "no versioned unsloth_zoo requirement at all, so pip may resolve a zoo without it"
    )
    floors = {}
    for req in reqs:
        lower = [
            Version(str(spec.version))
            for spec in req.specifier
            if spec.operator in (">=", "==", "~=")
        ]
        assert lower, f"unsloth_zoo requirement {req} has no lower bound"
        floors[str(req)] = max(lower)
    stale = {
        raw: str(floor)
        for raw, floor in floors.items()
        if floor < ZOO_FLOOR_WITH_THE_VISION_CHUNKER
    }
    assert not stale, (
        f"pyproject.toml still accepts unsloth_zoo {stale}, which predates "
        f"grpo_vision_chunks. Every vision GRPO batch raises on such an install, including "
        f"the single image grid models that used to work. Move the floor to "
        f"{ZOO_FLOOR_WITH_THE_VISION_CHUNKER} in the same commit that requires the helper."
    )
    assert len(set(floors.values())) == 1, (
        f"pyproject.toml declares more than one unsloth_zoo floor: {floors}. One of them is "
        f"the one users actually resolve."
    )


def test_the_upgrade_message_names_the_release_the_floor_names() -> None:
    """A message that says only "upgrade unsloth_zoo" leaves the user guessing which one."""
    source = RL_REPLACEMENTS.read_text(encoding = "utf-8")
    lines = source.splitlines()
    raised = [
        "\n".join(lines[i : i + 5])
        for i, line in enumerate(lines)
        if "Unsloth: vision GRPO needs an unsloth_zoo build that exports" in line
    ]
    assert len(raised) == 2, f"expected the no-grad and the gradient gate, found {len(raised)}"
    for block in raised:
        assert str(ZOO_FLOOR_WITH_THE_VISION_CHUNKER) in block, block
