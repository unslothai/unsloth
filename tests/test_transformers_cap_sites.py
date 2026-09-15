# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The published version window and every CI lane that mirrors it have to agree.

`transformers<=5.5.0` was not one line. It was two lines in pyproject.toml, two more in
unsloth_zoo's, a `<5.5` in studio-backend-ci.yml, a `<5.6` in notebooks-ci.yml and
version-compat-ci.yml, and a policy gate in unsloth-zoo's consolidated-tests-ci.yml. When
the cap moved, any of those left behind would have kept CI testing a range users no longer
get, and nothing would have gone red: the lane still runs, still passes, and simply proves
the wrong thing. That is the failure this file exists for.

5.5.0 mattered because it is exactly where prequantized bnb-4bit checkpoints lose
`quant_state` on every `Linear4bit` (#9867, #10010, #10017, #10276), and because it puts the
Gemma 4 E4B LoRA fix one patch release out of reach: that shipped in 5.5.2 (#5355). So the
assertions are not "the number is 5.17.0" but "the window
admits what was measured, still rejects what was rejected, and no lane sits below it
without saying why".

Reads files only: no network, no torch, no transformers install. That is what lets this run
on Windows and macOS runners as well as Linux.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"
WORKFLOWS = REPO / ".github" / "workflows"

# The newest transformers the version matrix was run against, and so the ceiling
# pyproject.toml is allowed to declare. Moving this is a decision: the sweep has to be
# re-run on the new release first.
TESTED_CEILING = Version("5.17.0")

# Releases that were tested and rejected. Every one of these has to stay excluded; a
# rewrite of the specifier that drops one silently re-admits a broken release.
REJECTED = (
    "4.52.0", "4.52.1", "4.52.2", "4.52.3", "4.53.0", "4.54.0",
    "4.55.0", "4.55.1", "4.57.0", "4.57.4", "4.57.5", "5.0.0", "5.1.0",
)

# Releases inside the newly opened part of the window. Named rather than generated so the
# test keeps meaning something after the ceiling moves again.
NEWLY_ADMITTED = ("5.6.0", "5.10.1", "5.14.1", "5.15.1", "5.16.1", "5.17.0")

# Workflow lanes whose transformers requirement is deliberately NOT the published cap.
# Each needs a reason, because "it is lower" is otherwise indistinguishable from "it was
# forgotten", which is the whole bug this file is about.
PINNED_BY_DESIGN = {
    "version-compat-ci.yml": (
        "the floor lane pins transformers==4.57.6 on purpose: it is the oldest release "
        "the cap admits and the last 4.x, and it needs huggingface_hub < 1.0"
    ),
}

# unsloth_zoo bounds torch, and unsloth's own CPU lanes have to admit what that bound
# admits or they test a torch users cannot get. 2.14.0 is the newest release the matrix
# was run against.
TESTED_TORCH = Version("2.14.0")
TORCH_MIRROR_WORKFLOW = WORKFLOWS / "studio-export-capability-ci.yml"


def _toml() -> dict:
    """pyproject as a dict.

    tomllib is 3.11+, and requires-python here is >=3.9, so the import is lazy and the
    older interpreters skip rather than failing to collect. Same shape as
    _find_config in unsloth-zoo's tests/test_wheel_top_level_packages.py. The
    cap-site-consistency job in version-compat-ci.yml runs this on 3.12 across three
    operating systems, so the assertions are not left to a developer box.
    """
    if sys.version_info < (3, 11):
        pytest.skip("tomllib needs Python 3.11+")
    import tomllib

    return tomllib.loads(PYPROJECT.read_text(encoding = "utf-8"))


def _pyproject_transformers() -> list[Requirement]:
    data = _toml()
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
        if req.name.lower().replace("_", "-") == "transformers":
            out.append(req)
    return out


def _declared_window() -> SpecifierSet:
    reqs = _pyproject_transformers()
    assert reqs, "pyproject.toml declares no transformers requirement at all"
    windows = {str(req.specifier) for req in reqs}
    assert len(windows) == 1, (
        f"pyproject.toml declares {len(windows)} different transformers windows across its "
        f"extras: {sorted(windows)}. One of them will be the one users hit and the other "
        f"will be the one CI tests."
    )
    return SpecifierSet(windows.pop())


def _ceiling(window: SpecifierSet) -> Version:
    tops = [Version(str(spec.version)) for spec in window if spec.operator in ("<=", "<")]
    assert tops, f"the transformers window declares no upper bound at all: {window}"
    return max(tops)


def test_pyproject_declares_one_transformers_window() -> None:
    window = _declared_window()
    assert len(_pyproject_transformers()) >= 2, (
        "this test assumed pyproject names transformers in more than one place; if that "
        "stopped being true, the drift it guards against is gone and so is the point"
    )
    assert _ceiling(window) == TESTED_CEILING, (
        f"pyproject.toml caps transformers at {_ceiling(window)}, and the version matrix "
        f"was run against {TESTED_CEILING}. Raising the cap means running the sweep on the "
        f"new release and moving TESTED_CEILING here in the same commit."
    )


def test_the_window_admits_every_release_the_sweep_passed() -> None:
    window = _declared_window()
    missing = [v for v in NEWLY_ADMITTED if v not in window]
    assert not missing, (
        f"the transformers window {window} excludes {missing}, which the sweep passed. An "
        f"exclusion has to name the test that failed on that release."
    )


def test_every_rejected_release_is_still_rejected() -> None:
    window = _declared_window()
    readmitted = [v for v in REJECTED if v in window]
    assert not readmitted, (
        f"the transformers window {window} now admits {readmitted}, which were tested and "
        f"rejected. Rewriting the specifier must not drop an exclusion."
    )


def _workflow_transformers_specs(path: Path) -> list[tuple[str, Requirement]]:
    """Every `transformers<spec>` requirement spelled inside a workflow's shell steps."""
    text = path.read_text(encoding = "utf-8")
    out = []
    for match in re.finditer(r"['\"](transformers[<>=!,.\d\s]*)['\"]", text):
        try:
            req = Requirement(match.group(1))
        except InvalidRequirement:
            continue
        if req.name.lower() == "transformers" and str(req.specifier):
            out.append((match.group(1), req))
    return out


def test_no_workflow_lane_sits_below_the_declared_ceiling() -> None:
    ceiling = _ceiling(_declared_window())
    offenders = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for raw, req in _workflow_transformers_specs(path):
            # An exact pin is a deliberate point in the range, not a cap that drifted.
            if any(spec.operator == "==" for spec in req.specifier):
                continue
            if ceiling in req.specifier:
                continue
            offenders.setdefault(path.name, []).append(raw)
    unexplained = {k: v for k, v in offenders.items() if k not in PINNED_BY_DESIGN}
    assert not unexplained, (
        f"these workflow lanes cap transformers below the {ceiling} the package publishes, "
        f"so they test a range users do not get: {unexplained}. Either widen the lane or "
        f"add it to PINNED_BY_DESIGN with the reason."
    )


def test_the_torch_mirror_admits_what_unsloth_zoo_admits() -> None:
    text = TORCH_MIRROR_WORKFLOW.read_text(encoding = "utf-8")
    found = re.findall(r"['\"](torch[<>=!,.\d\s]*)['\"]", text)
    specs = []
    for raw in found:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        if req.name.lower() == "torch" and str(req.specifier):
            specs.append((raw, req.specifier))
    assert specs, f"{TORCH_MIRROR_WORKFLOW.name} no longer pins torch; retarget this test"
    too_low = [raw for raw, spec in specs if TESTED_TORCH not in spec]
    assert not too_low, (
        f"{TORCH_MIRROR_WORKFLOW.name} caps torch below {TESTED_TORCH}, which unsloth_zoo "
        f"now admits: {too_low}. The two repos would disagree about the supported torch."
    )


def test_the_checker_rejects_the_window_that_shipped_the_defect() -> None:
    """Negative control. Every assertion above is a "nothing found" shape, which is also
    what a checker that has quietly stopped checking reports."""
    shipped = SpecifierSet(
        "".join(f"!={v}," for v in REJECTED) + ">=4.51.3,<=5.5.0"
    )
    assert "5.5.0" in shipped
    assert "5.17.0" not in shipped, "the old window must not admit the release that fixes it"
    for rejected in REJECTED:
        assert rejected not in shipped
