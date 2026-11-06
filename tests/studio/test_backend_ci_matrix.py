# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend CI runs one leg on the ceiling; the declared floor is checked statically by a lint."""

import ast
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "studio-backend-ci.yml"
FLOOR_CHECK = REPO / "tests" / "test_python39_compatibility.py"
BACKEND = REPO / "studio" / "backend"


# Written down, not derived, so moving to 3.14 is a deliberate decision; the single leg is the
# newest for removals-and-deprecations coverage.
CEILING = "3.13"


def _legs() -> dict[str, str]:
    """Maps each matrix scope to its interpreter; counts are not asserted, since shards add entries."""
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    matrix = document["jobs"]["pytest"]["strategy"]["matrix"]
    entries = matrix.get("include")
    assert entries, f"the matrix no longer lists its legs by scope: {matrix!r}"
    legs = {}
    for entry in entries:
        scope, python = str(entry["scope"]), str(entry["python"])
        assert legs.setdefault(scope, python) == python, (
            f"scope {scope!r} runs on more than one interpreter, so scope no longer says "
            f"which one a leg is: {entries!r}"
        )
    return legs


def _declared_floor() -> tuple[int, ...]:
    """The floor the workflow declares, which is what the lint aims at."""
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    floor = (document.get("env") or {}).get("PYTHON_FLOOR")
    assert floor, (
        "the workflow declares no PYTHON_FLOOR. With one leg in the matrix there is "
        "nothing else for the floor lint to aim at, so it would check the ceiling "
        "against itself and pass on anything."
    )
    return _version(str(floor))


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def test_the_full_suite_runs_on_the_ceiling():
    """The full suite runs on `CEILING`, since removed stdlib APIs break on the newest interpreter first."""
    legs = _legs()
    assert legs.get("full") == CEILING, (
        f"the full suite runs on {legs.get('full')!r}, not {CEILING!r}. Moving it is a "
        f"decision worth making deliberately: update CEILING here in the same change, "
        f"and say why the new one is the version where removals land first."
    )


def test_the_pre_312_branches_are_still_executed_somewhere():
    """What the dropped legs actually took away, and where it went.

    Seven backend files carry a sys.version_info branch. The 3.10 ones were never
    straddled even by the old matrix, whose oldest leg was 3.10, so every leg took the
    same side of them. 3.14 is above every leg there has ever been. The pre-3.12 side is
    the only thing a 3.13-only matrix stops executing, so it keeps a leg of its own,
    running those files and nothing else.
    """
    legs = _legs()
    spot = legs.get("floor-spot-check")
    assert spot, (
        "the floor spot-check leg is gone. With it, nothing anywhere takes the pre-3.12 "
        "side of native_path_leases.py, third_party_source.py or the folder-permission "
        "check, on a pull request or on main."
    )
    assert _version(spot) < (3, 12), (
        f"the spot-check leg runs {spot}, which takes the >= 3.12 side, so it re-tests "
        f"what the full leg already covers and the older side is executed nowhere."
    )
    assert _version(spot) >= _declared_floor(), (
        f"the spot-check leg runs {spot}, below the declared floor. It should be the "
        f"NEWEST version that still takes the old side, so a failure is about the "
        f"boundary rather than about being old."
    )


def _floor_lint() -> Path:
    return REPO / "scripts" / "lint_backend_python_floor.py"


def test_the_floor_is_linted_on_every_pull_request():
    """The floor lint must run on every pull request, so the oldest interpreter is checked before merge."""
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
    # vermin's answer is the gate's verdict, so a release that changes it must not move the gate
    # on its own: 1.9.0 failed 79 unchanged files on every pull request the morning it shipped.
    unpinned = [
        line for line in installs if not re.search(r"\bvermin==\d+(?:\.\d+)*(?=\s|$)", line)
    ]
    assert not unpinned, (
        f"{trigger_lint.name} installs vermin without an exact pin: {unpinned}. A range or a "
        f"wildcard (vermin==1.*) is not one either. A new vermin "
        f"release then decides the floor check by itself; pin it and move the pin deliberately."
    )


def test_the_floor_lint_reads_stdlib_availability_not_just_syntax():
    """Floor lint must check stdlib availability, not just syntax: `anext` parses but fails when run."""
    text = _floor_lint().read_text(encoding = "utf-8")
    assert "vermin" in text, (
        "the floor lint no longer uses vermin. Whatever replaces it has to read stdlib "
        "API availability and not only syntax, or it stops covering the case it exists for"
    )
    assert "PYTHON_FLOOR" in text, (
        "the floor lint no longer reads PYTHON_FLOOR from the workflow, so the number it "
        "checks against and the number the project declares can drift apart silently. It "
        "must not go back to reading the matrix either: with one leg, that would aim the "
        "check at the ceiling and pass on anything."
    )


def _boundaries() -> dict[str, Path]:
    """Finds every sys.version_info comparison in backend source and tests; a parse runs neither side."""
    found: dict[str, Path] = {}
    for path in sorted(BACKEND.rglob("*.py")):
        if "vendor" in path.parts:
            continue
        text = path.read_text(encoding = "utf-8", errors = "replace")
        for match in re.finditer(r"version_info\s*[<>]=?\s*\((\d+),\s*(\d+)\)", text):
            found[f"{path.name}:{match.start()}"] = path
    return found


def _straddles(legs: list[str], boundary: tuple[int, ...]) -> bool:
    versions = [_version(leg) for leg in legs]
    return any(v < boundary for v in versions) and any(v >= boundary for v in versions)


def test_every_version_boundary_lives_in_a_file_the_floor_lint_covers():
    """Each file with a version branch must be covered by the floor lint, as its old side is never run."""
    import importlib.util

    lint = REPO / "scripts" / "lint_backend_python_floor.py"
    spec = importlib.util.spec_from_file_location("lint_backend_python_floor", lint)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scanned = {Path(name) for name in module.targets()}

    boundaries = _boundaries()
    assert boundaries, "no version comparisons found in the backend; the scan is wrong"
    uncovered = sorted({str(path) for path in boundaries.values() if path not in scanned})
    assert not uncovered, (
        f"these files branch on sys.version_info and are not scanned by the floor lint: "
        f"{uncovered}. Nothing executes the older side of those branches any more, so the "
        f"lint's view of the names they use is the only check left on them."
    )


def test_the_declared_floor_is_still_checked_statically():
    """The static floor check must pass the pyproject.toml floor to ast.parse as feature_version."""
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


def test_the_parsed_floor_is_at_or_below_the_declared_floor():
    """The parsed floor must be at or below the declared floor, or the versions between them go
    unchecked."""
    text = (REPO / "pyproject.toml").read_text(encoding = "utf-8")
    declared = re.search(r"^requires-python\s*=\s*[\"'][^\"']*>=\s*(\d+)\.(\d+)", text, re.M)
    assert declared, "no >= lower bound in requires-python"
    parsed = (int(declared.group(1)), int(declared.group(2)))
    assert parsed <= _declared_floor(), (
        f"pyproject.toml declares {parsed} but the workflow declares a floor of "
        f"{_declared_floor()}. The parse has to reach at least as low as the lint, or the "
        f"versions between them are checked by nothing at all."
    )


def test_backend_ci_still_runs_on_push_to_main():
    """Backend CI must still run on push to main: a pull request never tests the merged result."""
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    triggers = document.get(True) or document.get("on") or {}
    push = triggers.get("push") or {}
    assert "main" in (push.get("branches") or []), (
        "Backend CI no longer runs on push to main, so nothing tests the merged result "
        "of two pull requests that were each green on their own merge commit"
    )


def test_the_floor_lint_scans_the_tree_rather_than_a_list_of_packages():
    """The floor lint must scan the whole tree; a named package list silently missed shipped files."""
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
    assert not missed, (
        f"the floor lint does not scan "
        f"{sorted(str(p.relative_to(backend)) for p in missed)}. Those files ship, and a "
        f"pull request no longer runs them on the oldest interpreter, so nothing else "
        f"would notice a stdlib symbol from above the floor.\n"
        f"\n"
        f"No file-level exemption is allowed here, deliberately. A deliberate above-floor "
        f"call is suppressed at the SITE with `# novermin` and a reason, which leaves the "
        f"rest of its module checked. Dropping the whole file would leave everything else "
        f"in it unchecked forever, which is the package-allowlist mistake one level down."
    )
    assert (
        len(scanned) > 300
    ), f"the floor lint only found {len(scanned)} files; the scan is not reaching the tree"


def test_the_floor_lint_covers_every_tree_the_matrix_legs_run():
    """The floor lint must cover unsloth_cli too, which the dropped legs ran on the floor interpreter."""
    import importlib.util

    lint = REPO / "scripts" / "lint_backend_python_floor.py"
    spec = importlib.util.spec_from_file_location("lint_backend_python_floor", lint)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    workflow = WORKFLOW.read_text(encoding = "utf-8")
    scanned = [str(Path(name).relative_to(REPO).as_posix()) for name in module.targets()]
    for tree in ("studio/backend", "unsloth_cli"):
        assert (
            tree in workflow
        ), f"{tree} is no longer run by {WORKFLOW.name}; drop it from the lint's ROOTS too"
        assert any(name.startswith(tree + "/") for name in scanned), (
            f"{WORKFLOW.name} still executes {tree} on the matrix, but the floor lint does "
            f"not scan it, so a post-floor stdlib name there passes the pull request and "
            f"fails on the push to main."
        )


def test_the_floor_lint_covers_test_code_the_matrix_executes():
    """The floor lint must scan test code too, since the matrix executes tests/ on every leg."""
    import importlib.util

    lint = REPO / "scripts" / "lint_backend_python_floor.py"
    spec = importlib.util.spec_from_file_location("lint_backend_python_floor", lint)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    scanned = {Path(name).relative_to(REPO).as_posix() for name in module.targets()}
    for tree in ("studio/backend/tests", "unsloth_cli/tests"):
        on_disk = {
            path.relative_to(REPO).as_posix()
            for path in (REPO / tree).rglob("*.py")
            if not any(part in module.EXCLUDE_PARTS for part in path.parts)
        }
        assert on_disk, f"{tree} has no python files; this assertion would pass on nothing"
        missed = sorted(on_disk - scanned)
        assert not missed, (
            f"the floor lint does not scan {missed}. The matrix runs those files on every "
            f"leg, including the oldest, so an above-floor API in them fails on main."
        )
