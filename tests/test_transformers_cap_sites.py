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

import inspect
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

# The ceiling every unsloth_zoo up to and including 2026.9.5 publishes. pip intersects
# unsloth's window with the zoo's, so this is what decides whether the window above is
# what a user actually resolves.
ZOO_TRANSFORMERS_CEILING_BEFORE_THE_LIFT = Version("5.5.0")

# STILL DEFERRED. No zoo release carries the matching transformers ceiling: 2026.9.5, the
# newest published, still says `transformers<=5.5.0` because unslothai/unsloth-zoo#1227 is
# open. Naming anything above it in pyproject.toml is a floor no release satisfies, which
# makes unsloth uninstallable rather than merely under-delivered, so the floor stops at
# 2026.9.5 and this gate stays off. The trl half of the same coordination is NOT deferred
# any more: 2026.9.5 ships unslothai/unsloth-zoo#1260, so
# ZOO_FLOOR_WITH_LIFTED_TRL_CAP in tests/test_trl_cap_sites.py names it and that gate runs.
# Set this to the release that ships #1227 and raise the pyproject floor to match, in the
# same commit; the gate re-enables itself.
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
    a user installing any extra gets the LOWER of the two ceilings. unsloth_zoo 2026.9.5
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
            "(unslothai/unsloth-zoo#1227) is published yet, and 2026.9.5 is the newest on "
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
    has ever been. Both are true at 2026.9.5, so this keeps running while the gate above
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


def _matrix_module(urlopen):
    """`tests/version_compat/test_transformers_pinned_symbols.py`, imported fresh with
    `urllib.request.urlopen` replaced.

    Imported under its own name, because the matrix is built at import: the substitution
    has to be in place before the module body runs, and the real module may already be in
    `sys.modules` from a full-suite run.
    """
    import importlib.util
    import urllib.request

    path = (
        Path(__file__).resolve().parent / "version_compat" / "test_transformers_pinned_symbols.py"
    )
    spec = importlib.util.spec_from_file_location("_matrix_under_test", path)
    module = importlib.util.module_from_spec(spec)
    original = urllib.request.urlopen
    urllib.request.urlopen = urlopen
    try:
        spec.loader.exec_module(module)
    finally:
        urllib.request.urlopen = original
    return module


def test_a_pypi_outage_keeps_every_load_bearing_tag() -> None:
    """The fallback list holds one tag per minor, so it does not carry the anchors.

    `_ALWAYS` names the patches a specific check exists for: v5.5.0 is the Apple Silicon
    ceiling and v5.16.0 is the tokenizers breakpoint. Returning `_TAGS_FALLBACK` unmerged
    let a transient PyPI failure drop both and still report green on a smaller matrix.
    """
    import urllib.error

    def refuses(*args, **kwargs):
        raise urllib.error.URLError("pypi is unreachable")

    module = _matrix_module(refuses)

    assert set(module._ALWAYS).issubset(
        module.TRANSFORMERS_TAGS
    ), "a PyPI outage dropped a load-bearing tag from the matrix"
    # The frozen list is still the body of it, so the outage does not shrink coverage.
    assert set(module._TAGS_FALLBACK).issubset(module.TRANSFORMERS_TAGS)
    assert module.TRANSFORMERS_TAGS[-1] == "main"


def test_the_outage_fallback_reaches_the_declared_floor() -> None:
    """An outage must not quietly move the floor up.

    `_FLOOR` is derived from pyproject, so the live matrix starts where the claim does.
    The frozen fallback is a separate list and began at 4.57.6, so any PyPI failure
    dropped every 4.52 through 4.56 check and CI could go green straight through a
    regression at the newly supported low end. `_ALWAYS` does not restore them: it names
    the Apple Silicon ceiling and the tokenizers breakpoint, both 5.x.
    """
    import urllib.error

    def refuses(*args, **kwargs):
        raise urllib.error.URLError("pypi is unreachable")

    module = _matrix_module(refuses)
    declared = module._declared_floor()
    concrete = [tag for tag in module.TRANSFORMERS_TAGS if tag != "main"]
    assert concrete, "the outage fallback resolved no tags at all"
    lowest = min(module._sort_key(tag) for tag in concrete)
    assert lowest <= declared, (
        f"under a PyPI outage the oldest tag checked is {lowest}, above the declared "
        f"floor {declared}, so every release between them goes unchecked while CI is green"
    )
    # Every supported minor below the old 4.57.6 start, not just the floor itself.
    minors = {module._sort_key(tag)[:2] for tag in concrete}
    missing = [m for m in ((4, 52), (4, 53), (4, 54), (4, 55), (4, 56)) if m not in minors]
    assert not missing, f"the outage fallback covers no tag for supported minors {missing}"


def test_an_empty_release_index_keeps_every_load_bearing_tag() -> None:
    """NEGATIVE CONTROL for the other fallback: a reachable PyPI that yields no usable
    release takes a different return path, and it has to merge the anchors too."""
    import io
    import json as _json

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    def empty(*args, **kwargs):
        return _Response(_json.dumps({"releases": {}}).encode("utf-8"))

    module = _matrix_module(empty)

    assert set(module._ALWAYS).issubset(module.TRANSFORMERS_TAGS)


def test_the_declared_ceiling_stays_in_the_matrix_after_a_patch_release() -> None:
    """The matrix keeps one tag per minor, so a 5.17.1 would evict 5.17.0.

    5.17.0 is the exact maximum `transformers<=5.17.0` admits. Letting a later patch take
    its slot would stop checking the supported ceiling and start checking a version no
    user can resolve through the declared window. The anchor is derived from pyproject's
    own cap, so lifting the cap moves it rather than leaving a stale literal behind.
    """
    import io
    import json as _json

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    published = {
        "5.17.0": [{"yanked": False}],
        # The patch that has not shipped yet, which is what evicts the ceiling.
        "5.17.1": [{"yanked": False}],
    }

    def releases(*args, **kwargs):
        return _Response(_json.dumps({"releases": published}).encode("utf-8"))

    module = _matrix_module(releases)

    assert module._declared_ceiling_tag() == (
        "v5.17.0",
    ), "the anchor is no longer read from pyproject's transformers cap"
    assert "v5.17.1" in module.TRANSFORMERS_TAGS, "the newest patch is still measured"
    assert (
        "v5.17.0" in module.TRANSFORMERS_TAGS
    ), "a patch release evicted the declared ceiling from the matrix"


def test_the_matrix_is_shared_between_xdist_workers(tmp_path, monkeypatch) -> None:
    """Every xdist worker must collect the same parameters.

    Each worker imports the matrix module and resolves the matrix itself during collection,
    so four PyPI reads are four chances to disagree: one timing out while the others succeed
    gives that worker the fallback list, the parameter sets diverge and xdist aborts the run
    instead of executing the fallback matrix. With the cache path set, the first process to
    resolve publishes the answer and the rest read it.
    """
    import io
    import json as _json
    import urllib.error

    cache = tmp_path / "matrix.json"
    monkeypatch.setenv("PYTEST_TRANSFORMERS_MATRIX_FILE", str(cache))

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    def succeeds(*args, **kwargs):
        return _Response(_json.dumps({"releases": {"5.17.0": [{"yanked": False}]}}).encode("utf-8"))

    def refuses(*args, **kwargs):
        raise urllib.error.URLError("this worker's read timed out")

    first = _matrix_module(succeeds)
    assert cache.is_file(), "the resolved matrix was not published for the other workers"

    # The worker whose own read fails must still collect what the first one published,
    # rather than the fallback list.
    second = _matrix_module(refuses)
    assert (
        second.TRANSFORMERS_TAGS == first.TRANSFORMERS_TAGS
    ), "a failed read gave one worker a different matrix, which aborts an xdist run"
    assert "v5.17.0" in second.TRANSFORMERS_TAGS


def test_without_the_cache_a_failed_read_still_falls_back(tmp_path, monkeypatch) -> None:
    """NEGATIVE CONTROL: the cache is a sharing mechanism, not a new dependency. With no
    path set, a failed read still yields the frozen matrix rather than nothing."""
    import urllib.error

    monkeypatch.delenv("PYTEST_TRANSFORMERS_MATRIX_FILE", raising = False)

    def refuses(*args, **kwargs):
        raise urllib.error.URLError("pypi is unreachable")

    module = _matrix_module(refuses)
    assert set(module._TAGS_FALLBACK).issubset(module.TRANSFORMERS_TAGS)


def test_a_pinned_patch_release_is_not_evicted_by_a_later_one() -> None:
    """One tag per minor keeps the matrix bounded, but not at the cost of a pin users run.

    Several notebooks pin 5.10.1, which this repo's own NEWLY_ADMITTED list names and
    notebooks-ci.yml calls out. With 5.10.4 published, the (5, 10) slot becomes 5.10.4 and
    5.10.1 stops being checked, so a symbol Unsloth needs that only arrived in a later 5.10
    patch would leave those notebooks broken while this matrix stayed green.
    """
    import io
    import json as _json

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    def releases(*args, **kwargs):
        published = {version: [{"yanked": False}] for version in ("5.10.1", "5.10.4", "5.15.1")}
        return _Response(_json.dumps({"releases": published}).encode("utf-8"))

    module = _matrix_module(releases)

    assert (
        "v5.10.3" in module.TRANSFORMERS_TAGS
    ), "the newest 5.10 patch is still measured (PyPI 5.10.4 is tagged v5.10.3 upstream)"
    assert "v5.10.1" in module.TRANSFORMERS_TAGS, "a later patch evicted the pinned 5.10.1"
    assert "v5.15.1" in module.TRANSFORMERS_TAGS

    # The pins this repo names as newly admitted are the ones that must survive eviction.
    for version in ("5.10.1", "5.15.1"):
        assert (
            "v" + version in module._ALWAYS
        ), f"{version} is in NEWLY_ADMITTED but is not anchored in the matrix"


def test_an_import_lane_pins_the_declared_ceiling() -> None:
    """Something has to import the supported maximum, not just the supported minimum.

    The `latest` lane is unpinned, so the day upstream publishes above the cap it resolves
    a combination no user can install through the declared window, and the floor lane
    becomes the only import-time evidence for a supported one. The static symbol suite
    reads source and cannot see import-time breakage, so nothing else covers it.
    """
    import yaml

    workflow = yaml.safe_load((WORKFLOWS / "version-compat-ci.yml").read_text(encoding = "utf-8"))
    job = workflow["jobs"]["zoo-imports-under-spoof"]
    lanes = {lane["slug"]: lane for lane in job["strategy"]["matrix"]["include"]}

    assert (
        "ceiling" in lanes
    ), "no import lane pins the declared ceiling, so only the floor is exercised"
    pins = " ".join(lanes["ceiling"]["pkg_pins"].split())
    assert (
        f"'transformers=={TESTED_CEILING}'" in pins
    ), f"the ceiling lane does not pin transformers=={TESTED_CEILING}; it reads {pins}"

    # The canary is the lane allowed to resolve outside the window, and it is the only one.
    assert lanes["latest"].get("continue_on_error") is True
    for slug in ("floor", "ceiling"):
        assert (
            lanes[slug].get("continue_on_error") is None
        ), f"the {slug} lane is inside the declared window, so it must stay blocking"


def test_the_ceiling_lane_moves_with_the_declared_window() -> None:
    """NEGATIVE CONTROL: the pin is a literal in YAML, so it can go stale exactly the way
    a cap can. A ceiling lane left on an older release is a lane testing a version the
    window no longer tops out at."""
    import yaml

    workflow = yaml.safe_load((WORKFLOWS / "version-compat-ci.yml").read_text(encoding = "utf-8"))
    lanes = {
        lane["slug"]: lane
        for lane in workflow["jobs"]["zoo-imports-under-spoof"]["strategy"]["matrix"]["include"]
    }
    pinned = re.search(r"'transformers==([0-9][^']*)'", lanes["ceiling"]["pkg_pins"])
    assert pinned is not None
    assert Version(pinned.group(1)) == TESTED_CEILING, (
        f"the ceiling lane pins {pinned.group(1)} while the declared window tops out at "
        f"{TESTED_CEILING}"
    )


def test_no_two_import_lanes_mint_the_same_pip_cache_key() -> None:
    """Lanes in this job share a cache name and key-files, so the interpreter is all that
    separates their keys.

    The key is `pip-v2-<name>-<os>-<arch>-py<minor>-<hash>`. Two lanes on one interpreter
    resolve to one key while installing different dependency sets, so whichever saves
    first wins and the other lane re-downloads its wheels every run. The restore step
    cannot take `${{ matrix.slug }}`, since tests/studio/test_pip_cache_naming.py requires
    a literal lowercase name, so distinct interpreters are what keeps the lanes apart.
    """
    import yaml

    workflow = yaml.safe_load((WORKFLOWS / "version-compat-ci.yml").read_text(encoding = "utf-8"))
    lanes = workflow["jobs"]["zoo-imports-under-spoof"]["strategy"]["matrix"]["include"]
    interpreters = [lane["python"] for lane in lanes]
    assert len(interpreters) == len(set(interpreters)), (
        f"two import lanes share an interpreter and so share one pip cache key: "
        f"{[(lane['slug'], lane['python']) for lane in lanes]}"
    )


def test_a_published_matrix_still_carries_the_anchors(tmp_path, monkeypatch) -> None:
    """Sharing the matrix between workers must not become a second source of truth.

    The published file is how the workers agree on the PyPI half of the answer. It can
    still be written by a different revision, left over from an earlier run, or pointed at
    by hand, and returning it verbatim dropped the floor, the old ceiling, the notebook
    pins and the declared ceiling while the suite reported green. That is the failure the
    fallback merge exists to prevent, arriving through the cache instead.
    """
    import json as _json
    import urllib.error

    cache = tmp_path / "published.json"
    # What a stale or foreign writer can leave behind: a list with none of the anchors.
    cache.write_text(_json.dumps(["v5.17.0"]), encoding = "utf-8")
    monkeypatch.setenv("PYTEST_TRANSFORMERS_MATRIX_FILE", str(cache))

    def refuses(*args, **kwargs):
        raise urllib.error.URLError("this worker reads the published file, not PyPI")

    module = _matrix_module(refuses)

    assert set(module._ALWAYS).issubset(
        module.TRANSFORMERS_TAGS
    ), "the published matrix was returned without its anchors"
    assert module._declared_ceiling_tag()[0] in module.TRANSFORMERS_TAGS
    # What the file did carry is still honoured, so sharing still does its job.
    assert "v5.17.0" in module.TRANSFORMERS_TAGS


def test_the_declared_ceiling_anchor_uses_the_tag_upstream_pushed() -> None:
    """NEGATIVE CONTROL for the derivation: upstream does not always tag a release under
    its own name, which is why _TAG_OVERRIDES exists. A ceiling landing on such a release
    must resolve to the tag that was pushed, or every check fails on the fetch rather than
    on the symbol it meant to test."""
    import importlib.util

    path = (
        Path(__file__).resolve().parent / "version_compat" / "test_transformers_pinned_symbols.py"
    )
    spec = importlib.util.spec_from_file_location("_ceiling_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for release, tag in module._TAG_OVERRIDES.items():
        assert module._TAG_OVERRIDES.get(release) == tag
        # The derivation has to consult the same table the matrix does.
        assert "_TAG_OVERRIDES" in inspect.getsource(
            module._declared_ceiling_tag
        ), "the ceiling anchor is built as 'v' + version and ignores the override table"


# ---------------------------------------------------------------------------
# The deferral above has to expire by itself.
# ---------------------------------------------------------------------------


def _newest_published_zoo_transformers_ceiling(timeout: float = 10.0):
    """(zoo version, its transformers ceiling) for the newest unsloth_zoo on PyPI.

    Returns None when PyPI cannot be asked, or when the newest release declares no
    transformers upper bound this function can read. Never raises: the caller treats
    "could not ask" as "keep deferring", so an offline runner is not a failure.
    """
    import json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(
            "https://pypi.org/pypi/unsloth_zoo/json", timeout = timeout
        ) as handle:
            payload = json.load(handle)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None

    info = payload.get("info") or {}
    released = info.get("version")
    if not released:
        return None

    # The zoo splits transformers by platform marker, and ordinary (non Apple Silicon)
    # installs get the widest line. Whole SpecifierSets are kept rather than a bare ceiling
    # Version: `<5.17.0` and `<=5.17.0` name the same number and mean different things, and
    # collapsing them said a zoo declaring `<5.17.0` covered our `<=5.17.0` window and
    # expired the deferral on a release that still cannot resolve our ceiling.
    windows = []
    for raw in info.get("requires_dist") or []:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        if req.name.lower() != "transformers" or not req.specifier:
            continue
        windows.append(req.specifier)
    if not windows:
        return None
    return Version(released), windows


def test_the_zoo_deferral_expires_when_the_zoo_release_ships() -> None:
    """A deferral nothing can end is the defect this file exists to catch.

    `test_the_declared_zoo_floor_can_supply_the_declared_transformers_window` skips while
    `ZOO_FLOOR_WITH_LIFTED_TRANSFORMERS_CAP` is None, and that constant is hand-written.
    So on the day unslothai/unsloth-zoo#1227 ships, nothing turns red: the gate keeps
    skipping, the pyproject floor keeps naming a zoo that caps transformers at 5.5.0, and
    the lift stays advertised rather than delivered for as long as nobody happens to look.
    That is precisely "a site left behind after the window moves does not go red", which
    is the failure the rest of this file is about, reproduced inside its own deferral.

    So the deferral is made self-expiring. This asks PyPI what the newest published
    unsloth_zoo actually allows, and fails only on POSITIVE evidence that the deferral is
    obsolete. No network, a timeout, a malformed answer or a zoo with no readable ceiling
    all leave it skipped: it can only ever turn red by proving the release landed, never
    by failing to reach PyPI. That keeps the three-OS cap-site job honest when it runs
    offline, which is the property that let this suite go on that job in the first place.
    """
    if ZOO_FLOOR_WITH_LIFTED_TRANSFORMERS_CAP is not None:
        pytest.skip(
            "the floor already names a zoo release carrying the lift, so the deferral is "
            "over and the gate above is live"
        )

    published = _newest_published_zoo_transformers_ceiling()
    if published is None:
        pytest.skip("PyPI could not be asked for unsloth_zoo, so the deferral stands")

    zoo_version, zoo_windows = published
    declared = _ceiling(_declared_window())
    # Membership, not a number comparison: the question is whether the published zoo can
    # actually resolve the exact ceiling declared here, which `<5.17.0` cannot and
    # `<=5.17.0` can.
    covering = [str(window) for window in zoo_windows if window.contains(declared)]
    assert not covering, (
        f"unsloth_zoo {zoo_version} is published and admits transformers {declared} "
        f"({', '.join(covering)}), the exact ceiling declared here, so the deferral is "
        f"over. Set ZOO_FLOOR_WITH_LIFTED_TRANSFORMERS_CAP to {zoo_version} and raise the "
        f"unsloth_zoo floor in pyproject.toml to it in the same commit; that re-enables "
        f"test_the_declared_zoo_floor_can_supply_the_declared_transformers_window, which "
        f"is what actually checks the two windows agree."
    )
