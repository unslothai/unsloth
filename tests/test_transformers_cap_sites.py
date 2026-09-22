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

# Newest transformers the matrix was run against; moving it means re-running the sweep first.
TESTED_CEILING = Version("5.17.0")

# NOT 4.51.3, which never worked: peft declares no transformers floor of its own, and peft
# 0.18.0 imports `GradientCheckpointingLayer` from `transformers.modeling_layers` (first present
# in 4.52.0), so this bound is all that stops a clean resolve raising ModuleNotFoundError at
# `import unsloth`. 4.52.4 rather than 4.52.0 because 4.52.0-4.52.3 are rejected below.
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

# Lanes deliberately NOT on the published cap, or "lower" reads as "forgotten". Keyed on
# (workflow, exact requirement), never the workflow alone: a filename-level exemption blinds the
# scan to every OTHER requirement in that file. Empty on purpose; a placeholder pre-authorises.
PINNED_BY_DESIGN: dict[tuple[str, str], str] = {}

# pip intersects our window with the zoo's, so the zoo's ceiling decides what resolves.
ZOO_TRANSFORMERS_CEILING_BEFORE_THE_LIFT = Version("5.5.0")

# DEFERRED: unslothai/unsloth-zoo#1227 is unpublished, and naming an unpublished floor makes
# unsloth uninstallable. Set to the release shipping it and raise the pyproject floor together.
# The trl half is NOT deferred: 2026.9.5 ships zoo#1260, so ZOO_FLOOR_WITH_LIFTED_TRL_CAP runs.
ZOO_FLOOR_WITH_LIFTED_TRANSFORMERS_CAP = None

# CPU lanes must admit what the zoo's torch bound admits, or they test a torch nobody gets.
TESTED_TORCH = Version("2.14.0")
TORCH_MIRROR_WORKFLOW = WORKFLOWS / "studio-export-capability-ci.yml"


def _toml() -> dict:
    """pyproject as a dict; tomllib is 3.11+ while requires-python is >=3.9, so import lazily."""
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
    """The TIGHTEST upper bound: `max` reported 5.18.0 for `<=5.17.0,<=5.18.0`, which resolves at
    5.17.0, so the assertions passed on exactly the stale cap this file exists to catch.
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
    """pip intersects our requirement with the zoo's and the user gets the LOWER ceiling, and zoo
    2026.9.5 says `transformers<=5.5.0`. Deferred while the constant is None.
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
    """Two floors across the extras means one is what users hit and the other what CI reads; an
    unbounded `unsloth_zoo` admits every old release. Both hold at 2026.9.5.
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
    """`{job: pinned transformers}` per floor lane. Parsed, not grepped: the file also pins
    transformers exactly in lanes that are NOT the floor.
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
    """A floor rots downward as silently as a ceiling rots up: `transformers>=4.51.3` was declared
    beside `peft>=0.18.0`, a pair that cannot import, so the floor named an unrunnable combination.
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
    """Negative control: both checks above are equality checks, which a constant edited in the
    wrong direction satisfies too. 4.51.3 is the release that actually failed.
    """
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
    """A dormant exemption is a pre-authorisation: the day its string returns, the scan finds it and
    skips it, which is the failure this file exists to prevent.
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
    """Negative control: every assertion above is a "nothing found" shape, which is also what a
    checker that has stopped checking reports.
    """
    shipped = SpecifierSet("".join(f"!={v}," for v in REJECTED) + ">=4.51.3,<=5.5.0")
    assert "5.5.0" in shipped
    assert "5.17.0" not in shipped, "the old window must not admit the release that fixes it"
    for rejected in REJECTED:
        assert rejected not in shipped


def test_this_file_is_triggered_by_everything_it_scans() -> None:
    """A gate its own workflow's `paths:` filter cannot start is not a gate: the sweeps read every
    `.github/workflows/*.yml`, so moving a cap in one of them has to trigger this workflow.
    """
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
    """NEGATIVE CONTROL: keyed on the filename, one intentional pin exempted every other
    transformers requirement in that file, so a range cap could go stale in the scanned workflow.
    """
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
    """The pinned-symbols module, imported fresh with `urlopen` replaced, under its own name: the
    matrix is built at import, so the substitution must precede the module body.
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
    """`_ALWAYS` names the patches a check exists for (v5.5.0 Apple Silicon ceiling, v5.16.0
    tokenizers breakpoint); returning `_TAGS_FALLBACK` unmerged dropped both and reported green.
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
    """An outage must not quietly move the floor up: the frozen fallback began at 4.57.6, so any PyPI
    failure dropped every 4.52-4.56 check. `_ALWAYS` cannot restore them, both its anchors are 5.x.
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
    """NEGATIVE CONTROL: a reachable PyPI yielding no usable release takes a different return
    path, which has to merge the anchors too.
    """
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
    """One tag per minor means a 5.17.1 would evict 5.17.0, the exact maximum the window admits, and
    check a version no user can resolve instead. The anchor derives from pyproject's cap.
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
    """Each worker resolves the matrix during collection, so one timing out takes the fallback list,
    the parameter sets diverge and xdist aborts. With the cache set, the first to resolve publishes.
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

    # A worker whose own read fails must still collect what the first published.
    second = _matrix_module(refuses)
    assert (
        second.TRANSFORMERS_TAGS == first.TRANSFORMERS_TAGS
    ), "a failed read gave one worker a different matrix, which aborts an xdist run"
    assert "v5.17.0" in second.TRANSFORMERS_TAGS


def test_without_the_cache_a_failed_read_still_falls_back(tmp_path, monkeypatch) -> None:
    """NEGATIVE CONTROL: the cache is a sharing mechanism, not a new dependency."""
    import urllib.error

    monkeypatch.delenv("PYTEST_TRANSFORMERS_MATRIX_FILE", raising = False)

    def refuses(*args, **kwargs):
        raise urllib.error.URLError("pypi is unreachable")

    module = _matrix_module(refuses)
    assert set(module._TAGS_FALLBACK).issubset(module.TRANSFORMERS_TAGS)


def test_a_pinned_patch_release_is_not_evicted_by_a_later_one() -> None:
    """Notebooks pin 5.10.1; with 5.10.4 published the (5, 10) slot becomes 5.10.4, so a symbol
    arriving in a later 5.10 patch would break them while this matrix stayed green.
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
    """The `latest` lane is unpinned, so once upstream publishes above the cap it resolves a
    combination no user can install. The static symbol suite cannot see import-time breakage.
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
    """NEGATIVE CONTROL: the pin is a YAML literal, so it goes stale exactly the way a cap does."""
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
    """Two lanes on one interpreter share a key, so whichever saves first wins and the other
    re-downloads. The restore step cannot take `matrix.slug`: the name must be a literal lowercase.
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
    """The published file can come from another revision or an earlier run, and returning it verbatim
    dropped the floor, the old ceiling and the notebook pins while the suite reported green.
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
    """NEGATIVE CONTROL: upstream does not always tag a release under its own name, so a ceiling
    landing on one must resolve to the tag pushed or every check fails on the fetch.
    """
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


def _newest_published_zoo_transformers_ceiling(timeout: float = 10.0):
    """(zoo version, transformers ceiling) for the newest unsloth_zoo on PyPI, else None. Never
    raises: the caller treats "could not ask" as "keep deferring".
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

    # Whole SpecifierSets, not a bare ceiling Version: `<5.17.0` and `<=5.17.0` name the same
    # number and differ, and collapsing them expired the deferral on a zoo that cannot resolve us.
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
    """The gate above skips on a hand-written constant, so this asks PyPI what the newest zoo allows
    and fails only on POSITIVE evidence it is obsolete; an offline runner stays skipped.
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
    # Membership, not a number comparison: `<5.17.0` cannot resolve our ceiling, `<=5.17.0` can.
    covering = [str(window) for window in zoo_windows if window.contains(declared)]
    assert not covering, (
        f"unsloth_zoo {zoo_version} is published and admits transformers {declared} "
        f"({', '.join(covering)}), the exact ceiling declared here, so the deferral is "
        f"over. Set ZOO_FLOOR_WITH_LIFTED_TRANSFORMERS_CAP to {zoo_version} and raise the "
        f"unsloth_zoo floor in pyproject.toml to it in the same commit; that re-enables "
        f"test_the_declared_zoo_floor_can_supply_the_declared_transformers_window, which "
        f"is what actually checks the two windows agree."
    )
