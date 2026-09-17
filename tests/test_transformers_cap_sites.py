# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The published transformers window and every CI lane that mirrors it have to agree.

The cap is spelled in pyproject.toml, unsloth_zoo's pyproject, and several workflows. A site
left behind when the cap moves keeps CI green while testing a range users no longer get, so
the assertions are not "the number is 5.17.0" but "the window admits what was measured,
still rejects what was rejected, and no lane sits below it without saying why".

Reads files only, which is what lets it run on the Windows and macOS runners too.
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

# The newest transformers the version matrix was run against. Moving it means re-running
# the sweep on the new release first.
TESTED_CEILING = Version("5.17.0")

# The oldest transformers the floor lanes actually run, and the floor pyproject declares.
# It is NOT 4.51.3, which never worked: peft declares no transformers floor of its own (a
# bare `transformers` on 0.18.0 through 0.21.0), and peft 0.18.0 -- the peft floor this same
# file declares -- imports `GradientCheckpointingLayer` from `transformers.modeling_layers`
# at peft/tuners/lora/model.py:26, a module that first exists in transformers 4.52.0. Our
# bound is therefore the only thing standing between a user and a resolve that installs
# cleanly and then raises ModuleNotFoundError at `import unsloth`. 4.52.4 and not 4.52.0
# because 4.52.0 through 4.52.3 are rejected below.
TESTED_FLOOR = Version("4.52.4")

# Tested and rejected; a specifier rewrite that drops one silently re-admits a broken release.
REJECTED = (
    "4.52.0",
    "4.52.1",
    "4.52.2",
    "4.52.3",
    "4.53.0",
    "4.54.0",
    "4.55.0",
    "4.55.1",
    "4.57.0",
    "4.57.4",
    "4.57.5",
    "5.0.0",
    "5.1.0",
)

# Named rather than generated, so the test still means something after the ceiling moves.
NEWLY_ADMITTED = ("5.6.0", "5.10.1", "5.14.1", "5.15.1", "5.16.1", "5.17.0")

# Lanes deliberately NOT on the published cap. Each needs a reason, or "lower" is
# indistinguishable from "forgotten", which is the bug this file is about. Keyed on
# (workflow, exact requirement string), never the workflow alone: a filename-level
# exemption blinds the scan to every OTHER transformers requirement in that same file.
# Empty on purpose. A placeholder entry is not free: it pre-authorises the exact string it
# names, so reintroducing `transformers<=5.5.0` in that workflow would be found by the scan
# and then skipped by the exemption, and the gate would stay green with nobody writing down
# why. The test below keeps the dict honest by requiring every entry to match a requirement
# some lane actually spells today.
PINNED_BY_DESIGN: dict[tuple[str, str], str] = {}

# The ceiling every unsloth_zoo up to and including 2026.9.4 publishes. pip intersects
# unsloth's window with the zoo's, so this is what decides whether the window above is
# what a user actually resolves.
ZOO_TRANSFORMERS_CEILING_BEFORE_THE_LIFT = Version("5.5.0")

# DEFERRED. The zoo release carrying the matching transformers ceiling
# (unslothai/unsloth-zoo#1227) is not on PyPI: 2026.9.4 is the newest published release, so
# naming anything above it in pyproject.toml is a floor no release satisfies, which makes
# unsloth uninstallable rather than merely under-delivered. The floor therefore stays at
# 2026.9.4 and the gate below stays off. Set this to the release that ships #1227 and raise
# the pyproject floor to match, in the same commit; the gate re-enables itself.
ZOO_FLOOR_WITH_LIFTED_TRANSFORMERS_CAP = None

# unsloth's CPU lanes must admit what unsloth_zoo's torch bound admits, or they test a
# torch users cannot get. 2.14.0 is the newest release the matrix was run against.
TESTED_TORCH = Version("2.14.0")
TORCH_MIRROR_WORKFLOW = WORKFLOWS / "studio-export-capability-ci.yml"


def _toml() -> dict:
    """pyproject as a dict; tomllib is 3.11+ and requires-python is >=3.9, so lazy-import
    and skip rather than failing collection on the older interpreters."""
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
    """The TIGHTEST upper bound, which is the one that decides what resolves.

    `max` was wrong: raising a cap by adding a bound without removing the old one, as in
    `<=5.17.0,<=5.18.0`, still resolves at 5.17.0, and reporting 5.18.0 let the assertions
    below pass on exactly the stale cap this file exists to catch. At equal versions `<`
    excludes more than `<=`, so it wins the tie.
    """
    tops = [
        (Version(str(spec.version)), spec.operator)
        for spec in window
        if spec.operator in ("<=", "<")
    ]
    assert tops, f"the transformers window declares no upper bound at all: {window}"
    version, _operator = min(tops, key = lambda pair: (pair[0], pair[1] == "<="))
    return version


def _pyproject_zoo() -> list[Requirement]:
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
        if req.name.lower().replace("_", "-") == "unsloth-zoo" and req.specifier:
            out.append(req)
    return out


def test_the_declared_zoo_floor_can_supply_the_declared_transformers_window() -> None:
    """Widening the window here does nothing while the resolvable zoo still caps lower.

    unsloth_zoo publishes its own transformers requirement and pip intersects the two, so
    a user installing any extra gets the LOWER of the two ceilings. unsloth_zoo 2026.9.4
    on PyPI says `transformers<=5.5.0`, which is exactly the cap this PR lifts, so
    without a matching zoo floor the lift is advertised and not delivered: the bnb-4bit
    `quant_state` failures and the Gemma 4 E4B LoRA fix stay out of reach, and asking for
    a newly admitted transformers by hand is a resolver conflict rather than an install.

    DEFERRED while ZOO_FLOOR_WITH_LIFTED_TRANSFORMERS_CAP is None: see that constant. The
    body is kept rather than deleted so raising the floor later is one edit, and so this
    docstring stays as the written record of what the deferral costs.
    """
    if ZOO_FLOOR_WITH_LIFTED_TRANSFORMERS_CAP is None:
        pytest.skip(
            "deferred: no unsloth_zoo release carrying the lifted transformers ceiling "
            "(unslothai/unsloth-zoo#1227) is published yet, and 2026.9.4 is the newest on "
            "PyPI, so pyproject.toml holds the zoo floor there. Until it ships, the window "
            "this file checks is wider than what pip actually resolves. Re-enable by "
            "setting ZOO_FLOOR_WITH_LIFTED_TRANSFORMERS_CAP to the release carrying #1227."
        )
    ceiling = _ceiling(_declared_window())
    if ceiling <= ZOO_TRANSFORMERS_CEILING_BEFORE_THE_LIFT:
        pytest.skip(
            f"declared transformers ceiling {ceiling} is within what zoo "
            f"<{ZOO_FLOOR_WITH_LIFTED_TRANSFORMERS_CAP} already admits"
        )
    reqs = _pyproject_zoo()
    assert reqs, (
        "pyproject.toml declares a transformers ceiling above what the published "
        "unsloth_zoo admits, and names no versioned unsloth_zoo requirement at all"
    )
    floors = {}
    for req in reqs:
        lower = [
            Version(str(spec.version))
            for spec in req.specifier
            if spec.operator in (">=", "==", "~=")
        ]
        assert lower, (
            f"unsloth_zoo requirement {req} has no lower bound, so it admits the release "
            f"whose own transformers cap is {ZOO_TRANSFORMERS_CEILING_BEFORE_THE_LIFT}"
        )
        floors[str(req)] = max(lower)
    stale = {
        raw: str(floor)
        for raw, floor in floors.items()
        if floor < ZOO_FLOOR_WITH_LIFTED_TRANSFORMERS_CAP
    }
    assert not stale, (
        f"pyproject.toml admits transformers up to {ceiling} while still accepting "
        f"unsloth_zoo {stale}. pip intersects the two requirements, so users would "
        f"resolve the zoo that caps transformers at "
        f"{ZOO_TRANSFORMERS_CEILING_BEFORE_THE_LIFT} and the wider window here would "
        f"never take effect. Move the floor to "
        f"{ZOO_FLOOR_WITH_LIFTED_TRANSFORMERS_CAP} in the same commit as the ceiling."
    )
    assert len(set(floors.values())) == 1, (
        f"pyproject.toml declares more than one unsloth_zoo floor: {floors}. One of them "
        f"is the one users hit."
    )


def test_pyproject_declares_one_unsloth_zoo_floor() -> None:
    """The half of the check above that does not depend on which release the floor names.

    Whatever the floor is, there has to be exactly one of it and it has to be a lower
    bound. Two different floors across the extras means one is what users hit and the
    other is what CI reads, and an unbounded `unsloth_zoo` admits every old release there
    has ever been. Both are true at 2026.9.4, so this keeps running while the gate above
    is deferred, and it is what stops the deferral from silently costing all coverage.
    """
    reqs = _pyproject_zoo()
    assert reqs, "pyproject.toml names no versioned unsloth_zoo requirement at all"
    floors = {}
    for req in reqs:
        lower = [
            Version(str(spec.version))
            for spec in req.specifier
            if spec.operator in (">=", "==", "~=")
        ]
        assert lower, (
            f"unsloth_zoo requirement {req} has no lower bound, so it admits every zoo "
            f"release ever published, including those whose own transformers cap is "
            f"{ZOO_TRANSFORMERS_CEILING_BEFORE_THE_LIFT}"
        )
        floors[str(req)] = max(lower)
    assert len(set(floors.values())) == 1, (
        f"pyproject.toml declares more than one unsloth_zoo floor: {floors}. One of them "
        f"is the one users hit."
    )


def _floor(window: SpecifierSet) -> Version:
    bottoms = [Version(str(spec.version)) for spec in window if spec.operator in (">=", ">")]
    assert bottoms, f"the transformers window declares no lower bound at all: {window}"
    return max(bottoms)


def _floor_lane_transformers_pins() -> dict[str, str]:
    """`{job: pinned transformers}` for every version-compat lane whose slug is `floor`.

    Read out of the parsed YAML rather than grepped, because the file also pins transformers
    exactly in lanes that are NOT the floor (the pinned-symbol matrix), and a grep cannot
    tell those apart.
    """
    if sys.version_info < (3, 11):
        pytest.skip("yaml parsing here needs the 3.11+ interpreter the job uses")
    import yaml

    doc = yaml.safe_load((WORKFLOWS / "version-compat-ci.yml").read_text(encoding = "utf-8"))
    found = {}
    for job_name, job in (doc.get("jobs") or {}).items():
        for entry in ((job.get("strategy") or {}).get("matrix") or {}).get("include") or []:
            if entry.get("slug") != "floor":
                continue
            pins = " ".join(
                str(v) for k, v in entry.items() if k.endswith("pins") or k.endswith("pin")
            )
            match = re.search(r"transformers==([\d.]+)", pins)
            assert match, (
                f"the floor lane of {job_name} no longer pins transformers exactly, so "
                f"nothing makes it run the declared floor. Retarget this test or restore the pin."
            )
            found[job_name] = match.group(1)
    return found


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


def test_pyproject_declares_the_floor_the_lanes_run() -> None:
    """The ceiling half of this file has a twin: a floor can rot downward just as silently.

    It did. `transformers>=4.51.3` was declared alongside `peft>=0.18.0`, and that pair
    cannot import, so the published floor named a combination no user could run.
    """
    window = _declared_window()
    assert _floor(window) == TESTED_FLOOR, (
        f"pyproject.toml floors transformers at {_floor(window)} and the floor lanes run "
        f"{TESTED_FLOOR}. Lowering the floor means proving the lower release imports with "
        f"the peft floor declared beside it, in the same commit."
    )


def test_the_floor_lanes_run_the_declared_floor() -> None:
    lanes = _floor_lane_transformers_pins()
    assert len(lanes) >= 2, (
        f"expected both version-compat floor lanes to pin transformers; found {lanes}. A "
        f"floor nothing runs is the state this test exists to end."
    )
    drifted = {job: pin for job, pin in lanes.items() if Version(pin) != TESTED_FLOOR}
    assert not drifted, (
        f"these floor lanes pin a transformers other than the declared floor {TESTED_FLOOR}: "
        f"{drifted}. A lane above the floor leaves the floor untested; below it, the lane "
        f"tests a release users cannot install."
    )


def test_the_floor_excludes_the_release_that_could_not_import() -> None:
    """Negative control for the two above: both are equality checks, which a constant
    edited in the wrong direction satisfies just as well. 4.51.3 is the release that
    actually failed, so the window must refuse it however the floor is spelled."""
    window = _declared_window()
    assert "4.51.3" not in window, (
        "the transformers window admits 4.51.3 again. peft 0.18.0 imports "
        "transformers.modeling_layers, which that release does not have."
    )
    assert str(TESTED_FLOOR) in window, "the window must admit the floor it declares"


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
            if (path.name, raw.strip()) in PINNED_BY_DESIGN:
                continue
            offenders.setdefault(path.name, []).append(raw)
    unexplained = offenders
    assert not unexplained, (
        f"these workflow lanes cap transformers below the {ceiling} the package publishes, "
        f"so they test a range users do not get: {unexplained}. Either widen the lane or "
        f"add it to PINNED_BY_DESIGN with the reason."
    )


def test_every_exemption_names_a_lane_that_exists_today() -> None:
    """A dormant exemption is a pre-authorisation, not documentation.

    An entry for a requirement no lane spells cannot be checked by anything, and the day
    that string comes back the scan finds it and skips it, which is the failure this file
    exists to prevent. So an exemption has to describe a lane that is really there.
    """
    spelled = {
        (path.name, raw.strip())
        for path in sorted(WORKFLOWS.glob("*.yml"))
        for raw, _req in _workflow_transformers_specs(path)
    }
    dormant = sorted(key for key in PINNED_BY_DESIGN if key not in spelled)
    assert not dormant, (
        f"PINNED_BY_DESIGN exempts requirements no workflow spells today: {dormant}. "
        f"Each of those silently pre-authorises that exact string if it is reintroduced. "
        f"Delete the entry; add it back when a lane genuinely needs it."
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
    """Negative control: every assertion above is a "nothing found" shape, which is also
    what a checker that has stopped checking reports."""
    shipped = SpecifierSet("".join(f"!={v}," for v in REJECTED) + ">=4.51.3,<=5.5.0")
    assert "5.5.0" in shipped
    assert "5.17.0" not in shipped, "the old window must not admit the release that fixes it"
    for rejected in REJECTED:
        assert rejected not in shipped


def test_this_file_is_triggered_by_everything_it_scans() -> None:
    """A gate its own workflow's `paths:` filter cannot start is not a gate: the sweeps
    above read every `.github/workflows/*.yml`, so a PR that moves a cap in one of them and
    nothing else has to trigger this workflow."""
    if sys.version_info < (3, 11):
        pytest.skip("yaml parsing here needs the 3.11+ interpreter the job uses")
    import yaml

    workflow = yaml.safe_load((WORKFLOWS / "version-compat-ci.yml").read_text(encoding = "utf-8"))
    # PyYAML resolves a bare `on:` key to the boolean True (YAML 1.1), so read both.
    triggers = workflow.get("on", workflow.get(True)) or {}
    paths = (triggers.get("pull_request") or {}).get("paths") or []

    required = (".github/workflows/**", "tests/test_transformers_cap_sites.py")
    missing = [name for name in required if name not in paths]
    assert not missing, (
        f"version-compat-ci.yml's pull_request.paths does not cover {missing}, so a PR "
        f"touching only those files never starts the cap-site-consistency job that reads "
        f"them. Current filter: {paths}"
    )


def test_a_stale_range_cap_is_caught_even_in_an_allowlisted_workflow(tmp_path, monkeypatch) -> None:
    """NEGATIVE CONTROL for the exemption: keyed on the filename, one intentional pin
    exempted every other transformers requirement in that file, so a range cap could go
    stale in the very workflow this gate scans."""
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "version-compat-ci.yml").write_text(
        "run: |\n"
        "  pip install 'transformers==4.51.3'\n"  # exact pin: a point, not a cap
        "  pip install 'transformers>=4.51.3,<=5.5.0'\n",  # stale range cap: must be caught
        encoding = "utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "WORKFLOWS", workflows)

    with pytest.raises(AssertionError) as raised:
        test_no_workflow_lane_sits_below_the_declared_ceiling()

    assert "version-compat-ci.yml" in str(raised.value)
    assert "<=5.5.0" in str(raised.value)
    assert "==4.51.3" not in str(raised.value), "an exact pin is a point in the range, not a cap"
