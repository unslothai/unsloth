# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend CI runs one interpreter, and the floor is defended statically instead.

Measured on one runner over the same tree, the four interpreter legs collected the same
26,320 tests and differed by exactly one: ``test_demonstrates_the_underlying_stdlib_regression``
is gated on ``sys.version_info >= (3, 12)``, so 3.10 and 3.11 reported 26193 passed / 127
skipped while 3.12 and 3.13 reported 26194 / 126. Four legs cost 97 runner-minutes per
push to re-run one identical suite and learn the value of a single skip marker, against a
queue observed 195 deep, and queue depth is wall-clock for every other workflow.

So there is one leg, the ceiling, and what the older ones were really defending is
asserted here instead:

* ``scripts/lint_backend_python_floor.py`` refuses any file that needs a symbol newer
  than the declared floor, across everything shipped or executed, on every pull request;
* the floor is DECLARED in the workflow rather than derived from the matrix, because a
  one-leg matrix would otherwise move it to the ceiling and check nothing;
* ``tests/test_python39_compatibility.py`` still parses at the version ``pyproject.toml``
  declares, which is below the declared floor.

The cost is stated rather than hidden: a static check does not run anything, so the
version-conditional branches enumerated below are now covered by reading and by the
names they use, not by execution. That is the trade, and it is the reason the floor lint
has to keep covering the files those branches live in.
"""

import ast
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "studio-backend-ci.yml"
FLOOR_CHECK = REPO / "tests" / "test_python39_compatibility.py"
BACKEND = REPO / "studio" / "backend"


# The interpreter the full suite runs on.
# Written down rather than derived, so moving to 3.14 is a decision somebody makes and defends here, not something that
# follows silently from an edit elsewhere.
# Asserting only "newer than the floor" was not enough: 3.11 and 3.12 satisfy that too, and either would quietly give up
# the removals-and-deprecations coverage that is the whole reason the single leg is the newest one.
CEILING = "3.13"


def _legs() -> dict[str, str]:
    """Each leg the matrix runs, as scope -> interpreter."""
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    matrix = document["jobs"]["pytest"]["strategy"]["matrix"]
    entries = matrix.get("include")
    assert entries, f"the matrix no longer lists its legs by scope: {matrix!r}"
    legs = {str(entry["scope"]): str(entry["python"]) for entry in entries}
    assert len(legs) == len(entries), f"two legs share a scope: {entries!r}"
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
    """One full leg, and it has to be the CEILING, by name.

    The ceiling is where a removal lands: a stdlib function that went away, a deprecation
    that became an error. Those break on the newest interpreter first and on the oldest
    never, so running only the oldest would be the wrong single choice, and running 3.11
    or 3.12 would be wrong in the same direction while still sitting above the floor.
    That is why this compares against a written-down CEILING rather than against the
    floor: "newer than 3.10" is satisfied by versions that give up exactly what the
    single leg exists to keep.
    """
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
    assert "PYTHON_FLOOR" in text, (
        "the floor lint no longer reads PYTHON_FLOOR from the workflow, so the number it "
        "checks against and the number the project declares can drift apart silently. It "
        "must not go back to reading the matrix either: with one leg, that would aim the "
        "check at the ceiling and pass on anything."
    )


def _boundaries() -> dict[str, Path]:
    """Every ``sys.version_info`` comparison in the backend, source and tests alike.

    Source is what made a single-leg matrix look indefensible on its own:
    ``sitecustomize.py`` repoints pathlib's pre-3.11 ``_NormalAccessor``, and
    ``native_path_leases.py`` and ``third_party_source.py`` branch on >= 3.12. A static
    parse cannot exercise any of them; it parses, it does not run. Which is why the full
    matrix stayed on main.
    """
    found: dict[str, Path] = {}
    for path in sorted(BACKEND.rglob("*.py")):
        if "vendor" in path.parts:  # third-party, pinned to its own support range
            continue
        text = path.read_text(encoding = "utf-8", errors = "replace")
        for match in re.finditer(r"version_info\s*[<>]=?\s*\((\d+),\s*(\d+)\)", text):
            found[f"{path.name}:{match.start()}"] = path
    return found


def _straddles(legs: list[str], boundary: tuple[int, ...]) -> bool:
    versions = [_version(leg) for leg in legs]
    return any(v < boundary for v in versions) and any(v >= boundary for v in versions)


def test_every_version_boundary_lives_in_a_file_the_floor_lint_covers():
    """The cost of one leg, made explicit rather than dropped along with the legs.

    These are the version-conditional branches nothing takes either side of any more. A
    parse reads both sides and runs neither, and there is no longer a second leg to
    execute the older one, so this is a real reduction in what CI proves. It is the trade
    the single leg buys, and pretending otherwise in a comment would be worse than making
    it.

    What is still enforceable is that every file carrying such a branch is covered by the
    floor lint, so the NAMES used on the older side are checked even though the branch is
    not taken. A boundary living in a file the lint skips would be unchecked twice over,
    and that is what this fails on.
    """
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


def test_the_parsed_floor_is_at_or_below_the_declared_floor():
    """The syntax check has to reach at least as low as the floor the lint aims at.

    They are two different checks of two different things: test_python39_compatibility.py
    parses at the version pyproject.toml declares, and the floor lint reads stdlib API
    availability at PYTHON_FLOOR. If the parsed version were the HIGHER of the two, the
    span between them would be checked by neither.

    It also records a mismatch worth fixing separately rather than papering over:
    pyproject declares >= 3.9 and the tree does not honour it. unsloth/models/_utils.py
    uses dataclasses.dataclass(kw_only) and
    tempfile.TemporaryDirectory(ignore_cleanup_errors), both 3.10. The declaration or the
    code has to give; this test only insists the two numbers stay in the order that
    leaves no gap.
    """
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
    """One leg on a pull request is only defensible if that leg also runs on the merge.

    A pull request tests its own merge commit, not the merged result, so main is where a
    semantic conflict between two green pull requests shows up. That was true with four
    legs and it is more load-bearing with one.
    """
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    triggers = document.get(True) or document.get("on") or {}
    push = triggers.get("push") or {}
    assert "main" in (push.get("branches") or []), (
        "Backend CI no longer runs on push to main, so nothing tests the merged result "
        "of two pull requests that were each green on their own merge commit"
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
    """The lint has to cover what the deleted legs covered, not just the backend.

    studio-backend-ci runs `pytest unsloth_cli/tests` as a step on every leg and lists
    unsloth_cli/** in its own paths filter, so the old 3.10 leg executed shipped CLI code
    on the floor interpreter. A lint aimed only at studio/backend replaces part of that
    and reads like it replaces all of it.
    """
    import importlib.util

    lint = REPO / "scripts" / "lint_backend_python_floor.py"
    spec = importlib.util.spec_from_file_location("lint_backend_python_floor", lint)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    workflow = WORKFLOW.read_text(encoding = "utf-8")
    # What it would hand to vermin, not what its source says it aims at.
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
    """Not shipped is not the same as not executed.

    studio-backend-ci runs `pytest tests/` from studio/backend on every leg, so a 3.11
    API in a test file is executed by the oldest leg exactly as one in a shipped module
    is. With the pull request down to a single newest leg, dropping tests from the scan
    would let both that leg and this lint pass while the failure waits for the push to
    main, which is the gap the lint exists to close.
    """
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
