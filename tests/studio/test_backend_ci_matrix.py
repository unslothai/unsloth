# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend CI runs both ends of its matrix on a pull request, and all four on main.

Measured on one runner over the same tree, the four interpreter legs collect the same
26,320 tests and differ by exactly one: ``test_demonstrates_the_underlying_stdlib_regression``
is gated on ``sys.version_info >= (3, 12)``, so 3.10 and 3.11 report 26193 passed / 127
skipped while 3.12 and 3.13 report 26194 / 126.

Dropping the two interior legs rests on three things, all asserted here:

* both ENDS are kept. A single leg was tried and is wrong: the backend has
  version-conditional branches that only a run can exercise, so the floor has to execute;
* the syntax floor is still checked statically by ``tests/test_python39_compatibility.py``,
  at the version ``pyproject.toml`` declares, which is lower than any leg here;
* main still runs all four, so a difference peculiar to 3.11 or 3.12 is caught on merge
  rather than never.
"""

import ast
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "studio-backend-ci.yml"
FLOOR_CHECK = REPO / "tests" / "test_python39_compatibility.py"
BACKEND = REPO / "studio" / "backend"


def _matrices() -> tuple[list[str], list[str]]:
    """The pull-request list and the full list, read out of the matrix expression."""
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    expression = str(document["jobs"]["pytest"]["strategy"]["matrix"]["python"])
    lists = re.findall(r"fromJSON\('(\[[^']*\])'\)", expression)
    assert len(lists) == 2, f"expected a pull-request list and a full list, found {lists}"
    return tuple(ast.literal_eval(item) for item in lists)  # type: ignore[return-value]


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def test_the_pull_request_matrix_is_the_newest_leg():
    """One leg on a pull request, and it has to be the CEILING.

    The ceiling is where a removal lands: a stdlib function that went away, a
    deprecation that became an error. Those break on the newest interpreter first and on
    the oldest never, so running only the oldest would be the wrong single choice.

    The floor is given up here deliberately, and paid for in two other places, both
    asserted below: scripts/lint_backend_python_floor.py refuses source that needs more
    than the oldest leg, on every pull request; and main still runs all four, so anything
    only a run can find is caught at merge rather than never.
    """
    subset, full = _matrices()
    assert len(subset) == 1, (
        f"the pull-request matrix is {subset}. If a second leg is being added back, the "
        f"reason to do it is the one this file used to encode: a static check does not "
        f"execute a version-conditional branch. Say so here rather than leaving it bare."
    )
    assert set(subset) <= set(full), f"{subset} is not a subset of {full}"
    assert max(full, key = _version) == subset[0], (
        f"the pull-request leg is {subset[0]} but the newest is {max(full, key = _version)}. "
        f"A single leg has to be the newest: removals and deprecations land there first."
    )


def _floor_lint() -> Path:
    return REPO / "scripts" / "lint_backend_python_floor.py"


def test_the_floor_is_linted_on_every_pull_request():
    """What replaces the leg that was dropped, asserted through where it runs.

    Backend CI does not filter on .github/workflows/**, and more to the point a pull
    request that only touches backend source needs this to have run BEFORE the merge,
    which is the whole point of dropping the leg. workflow-trigger-lint.yml carries no
    paths filter at all, so it sees every pull request.
    """
    lint = _floor_lint()
    assert lint.is_file(), (
        f"{lint.name} is gone. It is the only thing checking the backend against the "
        f"oldest interpreter before a merge, now that a pull request runs only the newest."
    )
    trigger_lint = REPO / ".github" / "workflows" / "workflow-trigger-lint.yml"
    text = trigger_lint.read_text(encoding = "utf-8")
    assert lint.name in text, (
        f"{trigger_lint.name} no longer runs {lint.name}, so nothing checks the floor "
        f"before a merge"
    )
    installs = [line for line in text.splitlines() if "pip install" in line and "vermin" in line]
    assert installs, (
        f"{trigger_lint.name} does not pip install vermin, so {lint.name} exits with its "
        f"'not installed' message rather than checking anything. Asserted against the "
        f"install line rather than the file, because the first version of this check "
        f"looked for 'vermin' anywhere and was satisfied by a comment mentioning it."
    )


def test_the_floor_lint_reads_stdlib_availability_not_just_syntax():
    """Asserted through what it DOES, because the distinction is the reason it exists.

    ast.parse at a feature_version answers a syntax question. The regression that a
    dropped interpreter leg actually stops catching is a stdlib name that does not exist
    yet: core/research_runs.py already uses `anext`, which is 3.10, and that parses on
    every version and fails only when the line runs.
    """
    text = _floor_lint().read_text(encoding = "utf-8")
    assert "vermin" in text, (
        "the floor lint no longer uses vermin. Whatever replaces it has to read stdlib "
        "API availability and not only syntax, or it stops covering the case it exists for"
    )
    assert "fromJSON" in text, (
        "the floor lint no longer reads its target from the workflow matrix, so raising "
        "the matrix floor would silently leave it checking the old one"
    )


def _boundaries() -> dict[str, tuple[int, ...]]:
    """Every ``sys.version_info`` comparison in the backend, source and tests alike.

    Source is what made a single-leg matrix look indefensible on its own:
    ``sitecustomize.py`` repoints pathlib's pre-3.11 ``_NormalAccessor``, and
    ``native_path_leases.py`` and ``third_party_source.py`` branch on >= 3.12. A static
    parse cannot exercise any of them; it parses, it does not run. Which is why the full
    matrix stayed on main.
    """
    found: dict[str, tuple[int, ...]] = {}
    for path in sorted(BACKEND.rglob("*.py")):
        if "vendor" in path.parts:  # third-party, pinned to its own support range
            continue
        text = path.read_text(encoding = "utf-8", errors = "replace")
        for match in re.finditer(r"version_info\s*[<>]=?\s*\((\d+),\s*(\d+)\)", text):
            found[f"{path.name}:{match.start()}"] = (int(match.group(1)), int(match.group(2)))
    return found


def _straddles(legs: list[str], boundary: tuple[int, ...]) -> bool:
    versions = [_version(leg) for leg in legs]
    return any(v < boundary for v in versions) and any(v >= boundary for v in versions)


def test_the_boundaries_the_subset_stops_executing_are_still_run_on_main():
    """The trade, made explicit rather than dropped along with the legs.

    These are the version-conditional branches a pull request no longer takes either side
    of. Nothing static covers them -- a parse reads both sides and runs neither -- so the
    only thing that does is the full matrix on main. This lists them so the cost is
    visible, and fails if main stops covering them.
    """
    subset, full = _matrices()
    boundaries = _boundaries()
    assert boundaries, "no version comparisons found in the backend; the scan is wrong"
    lost = {
        where: boundary
        for where, boundary in boundaries.items()
        if _straddles(full, boundary) and not _straddles(subset, boundary)
    }
    assert lost, (
        "no boundary is lost by the single-leg subset, which would mean the scan found "
        "nothing; it is the reason this file explains the trade at all"
    )
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    triggers = document.get(True) or document.get("on") or {}
    push = triggers.get("push") or {}
    assert "main" in (push.get("branches") or []), (
        f"Backend CI no longer runs on push to main, so {sorted(lost)} are executed on "
        f"neither side of their boundary anywhere, and dropping the interior legs stops "
        f"being a trade and becomes a straight loss"
    )


def test_the_declared_floor_is_still_checked_statically():
    """The static check is what replaced the older legs, so it has to still be there.

    Asserted through what it DOES rather than by filename: it must read the floor from
    pyproject.toml and hand it to ast.parse as feature_version, which is what makes it a
    floor check rather than a parse on whatever interpreter happens to run it.
    """
    assert FLOOR_CHECK.is_file(), (
        f"{FLOOR_CHECK.name} is gone. It is what covers the declared floor, which is below "
        f"every leg this matrix runs, so removing it leaves that floor untested."
    )
    text = FLOOR_CHECK.read_text(encoding = "utf-8")
    assert "requires-python" in text, "the floor is no longer read from pyproject.toml"
    assert "feature_version" in text, (
        "the check no longer parses at the declared floor, so it would pass on syntax that "
        "the floor cannot parse"
    )


def test_the_static_floor_is_below_the_matrix_floor():
    """Otherwise the static check is not covering the versions the matrix stopped running."""
    text = (REPO / "pyproject.toml").read_text(encoding = "utf-8")
    declared = re.search(r"^requires-python\s*=\s*[\"'][^\"']*>=\s*(\d+)\.(\d+)", text, re.M)
    assert declared, "no >= lower bound in requires-python"
    floor = (int(declared.group(1)), int(declared.group(2)))
    _subset, full = _matrices()
    assert floor <= min(_version(leg) for leg in full), (
        f"the declared floor {floor} is above the oldest matrix leg, so the static check "
        f"is not covering what the matrix stopped running"
    )


def test_the_full_matrix_still_runs_on_main():
    """Dropping the interior legs is only defensible because main runs everything."""
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    triggers = document.get(True) or document.get("on") or {}
    push = triggers.get("push") or {}
    assert "main" in (push.get("branches") or []), (
        "Backend CI no longer runs on push to main, so the interpreter versions dropped "
        "from the pull-request matrix would not be tested anywhere"
    )


def test_the_floor_lint_scans_the_tree_rather_than_a_list_of_packages():
    """The shape of the first version of that lint, which is why this exists.

    It named core, utils and routes and silently missed 116 shipped files: all of hub,
    plugins, models, storage, auth, picker and state, plus _platform_compat.py, which
    main.py imports directly. It also named "loggers.py", which is a directory, so that
    entry matched nothing. A check covering most of a tree reads exactly like one
    covering all of it, and with the floor leg dropped this is the only thing looking.

    Asserted by counting what the lint would actually hand to vermin against what is on
    disk, rather than by reading its source for a glob.
    """
    import importlib.util

    lint = REPO / "scripts" / "lint_backend_python_floor.py"
    spec = importlib.util.spec_from_file_location("lint_backend_python_floor", lint)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    backend = REPO / "studio" / "backend"
    on_disk = {
        path
        for path in backend.rglob("*.py")
        if not any(part in module.EXCLUDE_PARTS for part in path.relative_to(backend).parts)
    }
    scanned = {Path(name) for name in module.targets()}
    missed = {p for p in on_disk if p not in scanned}
    # Only the recorded exemptions may be absent, and each carries its reason.
    allowed = {backend / name for name in module.GUARDED}
    assert missed <= allowed, (
        f"the floor lint does not scan {sorted(str(p.relative_to(backend)) for p in missed - allowed)}. "
        f"Those files ship, and a pull request no longer runs them on the oldest "
        f"interpreter, so nothing else would notice a stdlib symbol from above the floor."
    )
    assert len(scanned) > 300, (
        f"the floor lint only found {len(scanned)} files; the scan is not reaching the tree"
    )
