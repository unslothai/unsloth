# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The published TRL window, every CI lane mirroring it, and every TRL-keyed runtime
guard have to agree. A site left behind when the cap moves keeps CI green while testing
a range users no longer get, so nothing here asserts "the number is 1.13.0".

Guard reachability is what made the old `<=0.24.0` cap visible: `rl_replacements.py`
gates `openenv_vllm_reload_weights` on TRL >= 0.26.0 and `vllm_generation_init_patch` on
>= 0.28.0, so under that window unsloth shipped two patches no resolvable install could
reach and nothing failed.

Reads files only, which is what lets it run on the Windows and macOS runners.
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
# Restates pyproject's window verbatim: a second copy that can go stale on its own.
RUNTIME_MIRROR = REPO / "studio" / "backend" / "requirements" / "no-torch-runtime.txt"
RL_REPLACEMENTS = REPO / "unsloth" / "models" / "rl_replacements.py"

# Newest TRL the matrix was run against; moving it means re-running the sweep first.
TESTED_CEILING = Version("1.13.0")

# Tested and rejected; a specifier rewrite dropping one silently re-admits it.
REJECTED = ("0.19.0",)

# Releases the sweep actually ran, named not generated: 0.29.1 last 0.x, 1.0.0 first
# major, 1.7.0 the chunked_nll default flip (trl#5846), 1.13.0 the ceiling.
NEWLY_ADMITTED = ("0.29.1", "1.0.0", "1.6.0", "1.7.0", "1.13.0")

# pip intersects our window with the zoo's, so the zoo ceiling decides what resolves.
ZOO_TRL_CEILING_BEFORE_THE_LIFT = Version("0.24.0")

# Published and says `trl<=1.13.0` (unslothai/unsloth-zoo#1260), so the gate below is
# live. The transformers half stays deferred; one floor serves both.
ZOO_FLOOR_WITH_LIFTED_TRL_CAP = Version("2026.9.5")

# Lanes deliberately off the cap; without a reason "lower" reads as "forgotten". Keyed
# on (workflow, exact requirement), never filename: that would exempt the whole file.
PINNED_BY_DESIGN = {
    ("consolidated-tests-ci.yml", "trl>=0.18.2,<1.0.0"): (
        "the TRL<1 half of a deliberate two-lane split; the sibling lane is "
        "'trl>=1,<2', so between them the pair covers both majors and neither is a cap "
        "that drifted"
    ),
}


def _toml() -> dict:
    """tomllib is 3.11+ and requires-python is >=3.9: lazy-import and skip, never fail
    collection on an older interpreter."""
    if sys.version_info < (3, 11):
        pytest.skip("tomllib needs Python 3.11+")
    import tomllib

    return tomllib.loads(PYPROJECT.read_text(encoding = "utf-8"))


def _pyproject_trl() -> list[Requirement]:
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
        if req.name.lower().replace("_", "-") == "trl":
            out.append(req)
    return out


def _declared_window() -> SpecifierSet:
    reqs = _pyproject_trl()
    assert reqs, "pyproject.toml declares no trl requirement at all"
    windows = {str(req.specifier) for req in reqs}
    assert len(windows) == 1, (
        f"pyproject.toml declares {len(windows)} different trl windows across its extras: "
        f"{sorted(windows)}. One of them will be the one users hit and the other will be "
        f"the one CI tests."
    )
    return SpecifierSet(windows.pop())


def _ceiling(window: SpecifierSet) -> Version:
    """The TIGHTEST upper bound, since that is what decides what resolves.

    `max` is WRONG: `<=0.24.0,<=1.13.0` still resolves at 0.24.0, so reporting 1.13.0 let
    the assertions pass on exactly the stale cap this file exists to catch. `<` wins ties.
    """
    tops = [
        (Version(str(spec.version)), spec.operator)
        for spec in window
        if spec.operator in ("<=", "<")
    ]
    assert tops, f"the trl window declares no upper bound at all: {window}"
    version, _operator = min(tops, key = lambda pair: (pair[0], pair[1] == "<="))
    return version


def test_pyproject_declares_one_trl_window() -> None:
    window = _declared_window()
    assert len(_pyproject_trl()) >= 2, (
        "this test assumed pyproject names trl in more than one place; if that stopped "
        "being true, the drift it guards against is gone and so is the point"
    )
    assert _ceiling(window) == TESTED_CEILING, (
        f"pyproject.toml caps trl at {_ceiling(window)}, and the version matrix was run "
        f"against {TESTED_CEILING}. Raising the cap means running the sweep on the new "
        f"release and moving TESTED_CEILING here in the same commit."
    )


def test_the_window_admits_every_release_the_sweep_passed() -> None:
    window = _declared_window()
    missing = [v for v in NEWLY_ADMITTED if v not in window]
    assert not missing, (
        f"the trl window {window} excludes {missing}, which the sweep passed. An "
        f"exclusion has to name the test that failed on that release."
    )


def test_every_rejected_release_is_still_rejected() -> None:
    window = _declared_window()
    readmitted = [v for v in REJECTED if v in window]
    assert not readmitted, (
        f"the trl window {window} now admits {readmitted}, which were tested and "
        f"rejected. Rewriting the specifier must not drop an exclusion."
    )


def test_the_runtime_requirements_mirror_restates_the_same_window() -> None:
    """`no-torch-runtime.txt` copies pyproject's spec rather than resolving it."""
    window = _declared_window()
    found = []
    for line in RUNTIME_MIRROR.read_text(encoding = "utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            req = Requirement(line)
        except InvalidRequirement:
            continue
        if req.name.lower() == "trl":
            found.append(req.specifier)
    assert found, f"{RUNTIME_MIRROR.name} no longer names trl; retarget this test"
    disagree = [str(spec) for spec in found if str(spec) != str(window)]
    assert not disagree, (
        f"{RUNTIME_MIRROR.name} declares {disagree} while pyproject.toml declares "
        f"{window}. The installer would ship a different TRL range than the package "
        f"advertises."
    )


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


def test_the_declared_zoo_floor_can_supply_the_declared_trl_window() -> None:
    """Widening the window does nothing while the resolvable zoo still caps TRL lower.

    pip gives a user the LOWER of the two ceilings, and every zoo up to 2026.9.4 says
    `trl<=0.24.0`, under which the guards above stay unreachable. So the lift is only
    real once the zoo floor names a release carrying it; 2026.9.5 is that release.
    """
    ceiling = _ceiling(_declared_window())
    if ceiling <= ZOO_TRL_CEILING_BEFORE_THE_LIFT:
        pytest.skip(
            f"declared trl ceiling {ceiling} is within what zoo "
            f"<{ZOO_FLOOR_WITH_LIFTED_TRL_CAP} already admits; nothing to coordinate"
        )
    reqs = _pyproject_zoo()
    assert reqs, (
        "pyproject.toml declares a trl ceiling above what the published unsloth_zoo "
        "admits, and names no versioned unsloth_zoo requirement at all, so nothing "
        "stops pip resolving the zoo that caps trl lower"
    )
    floors = {}
    for req in reqs:
        lower = [
            Version(str(spec.version))
            for spec in req.specifier
            if spec.operator in (">=", "==", "~=")
        ]
        assert lower, (
            f"unsloth_zoo requirement {req} has no lower bound, so it admits the "
            f"release whose own trl cap is {ZOO_TRL_CEILING_BEFORE_THE_LIFT}"
        )
        floors[str(req)] = max(lower)
    stale = {
        raw: str(floor) for raw, floor in floors.items() if floor < ZOO_FLOOR_WITH_LIFTED_TRL_CAP
    }
    assert not stale, (
        f"pyproject.toml admits trl up to {ceiling} while still accepting unsloth_zoo "
        f"{stale}. pip intersects the two requirements, so users would resolve the zoo "
        f"that caps trl at {ZOO_TRL_CEILING_BEFORE_THE_LIFT} and the wider window here "
        f"would never take effect. Move the floor to "
        f"{ZOO_FLOOR_WITH_LIFTED_TRL_CAP} in the same commit as the ceiling."
    )
    assert len(set(floors.values())) == 1, (
        f"pyproject.toml declares more than one unsloth_zoo floor: {floors}. One of them "
        f"is the one users hit."
    )


def _trl_version_guards() -> list[tuple[int, Version]]:
    """Every `Version(importlib_version("trl")) < Version("X")` early-return in
    rl_replacements.py, as (line number, X)."""
    text = RL_REPLACEMENTS.read_text(encoding = "utf-8")
    out = []
    pattern = re.compile(
        r"""Version\(\s*importlib_version\(\s*["']trl["']\s*\)\s*\)\s*<\s*Version\(\s*["']([\d.]+)["']\s*\)"""
    )
    for match in pattern.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        out.append((line, Version(match.group(1))))
    return out


def test_no_trl_runtime_guard_is_unreachable_through_the_declared_window() -> None:
    """A patch gated on a TRL newer than the cap admits is dead code that ships green.

    This is the assertion the `<=0.24.0` cap failed: `openenv_vllm_reload_weights`
    (>= 0.26.0) and `vllm_generation_init_patch` (>= 0.28.0) both returned early on
    every install a user could resolve.
    """
    window = _declared_window()
    guards = _trl_version_guards()
    assert guards, (
        f"{RL_REPLACEMENTS.name} no longer gates any patch on a trl version; retarget "
        f"this test rather than deleting it"
    )
    unreachable = [(line, str(floor)) for line, floor in guards if not window.contains(str(floor))]
    assert not unreachable, (
        f"these patches in unsloth/models/rl_replacements.py require a TRL the declared "
        f"window {window} cannot resolve, so they can never run: {unreachable}. Either "
        f"lift the cap to admit that floor or delete the patch."
    )


def _workflow_trl_specs(path: Path) -> list[tuple[str, Requirement]]:
    """Every `trl<spec>` requirement spelled inside a workflow's shell steps."""
    text = path.read_text(encoding = "utf-8")
    out = []
    for match in re.finditer(r"""['"](trl[<>=!,.\d\s]*)['"]""", text):
        try:
            req = Requirement(match.group(1))
        except InvalidRequirement:
            continue
        if req.name.lower() == "trl" and str(req.specifier):
            out.append((match.group(1), req))
    return out


def test_no_workflow_lane_sits_below_the_declared_ceiling() -> None:
    ceiling = _ceiling(_declared_window())
    offenders: dict[str, list[str]] = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for raw, req in _workflow_trl_specs(path):
            # An exact pin is a deliberate point in the range, not a cap that drifted.
            if any(spec.operator == "==" for spec in req.specifier):
                continue
            if ceiling in req.specifier:
                continue
            if (path.name, raw.strip()) in PINNED_BY_DESIGN:
                continue
            offenders.setdefault(path.name, []).append(raw)
    assert not offenders, (
        f"these workflow lanes cap trl below the {ceiling} the package publishes, so they "
        f"test a range users do not get: {offenders}. Either widen the lane or add it to "
        f"PINNED_BY_DESIGN with the reason."
    )


def _blocking_trl_lanes(path: Path) -> list[tuple[str, str, str]]:
    """Every (job, lane, trl spec) in a workflow that can actually fail the run.

    Shape-agnostic on purpose. The lane that installs the newest TRL has been, at
    different points on this stack, an explicit `trl==<ceiling>` pin and an unpinned
    `latest` lane, and it has been both blocking and `continue-on-error`. What has to
    stay true is the property, not the spelling, so this reads the YAML and asks which
    lanes are blocking rather than grepping for a literal.
    """
    if sys.version_info < (3, 11):
        pytest.skip("yaml parsing here needs the 3.11+ interpreter the job uses")
    import yaml

    workflow = yaml.safe_load(path.read_text(encoding = "utf-8")) or {}
    out: list[tuple[str, str, str]] = []
    for job_name, job in (workflow.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        # `continue-on-error` in any form means this job is not a guaranteed gate; an
        # expression we cannot evaluate counts as not-blocking, which is the safe answer.
        if job.get("continue-on-error") not in (None, False):
            continue
        includes = (((job.get("strategy") or {}).get("matrix") or {}).get("include")) or []
        lanes = includes if isinstance(includes, list) and includes else [{}]
        for lane in lanes:
            if not isinstance(lane, dict):
                continue
            if lane.get("continue_on_error") not in (None, False):
                continue
            slug = str(lane.get("slug", "<no matrix>"))
            blob = " ".join(str(v) for v in lane.values())
            for token in re.finditer(r"""(?:^|['"\s])(trl(?:[<>=!,.\d]*)?)(?=['"\s]|$)""", blob):
                out.append((str(job_name), slug, token.group(1)))
    return out


def test_a_blocking_lane_installs_a_trl_from_the_newly_admitted_major() -> None:
    """A TRL 1.x regression has to be able to fail this workflow.

    Satisfied either by a blocking lane that pins an explicit 1.x inside the window, or
    by a blocking lane that leaves trl unpinned and therefore resolves PyPI's newest. It
    is NOT satisfied when the only lane reaching 1.x is `continue-on-error`, which is the
    state this test exists to reject: lifting the cap while the sole 1.x lane is a canary
    would advertise a range no gate defends.
    """
    window = _declared_window()
    lanes = _blocking_trl_lanes(WORKFLOWS / "version-compat-ci.yml")
    assert lanes, (
        "version-compat-ci.yml has no blocking lane that installs trl at all; retarget "
        "this test rather than deleting it"
    )
    reaches_one_x = []
    for job, slug, spec in lanes:
        if spec == "trl":  # unpinned: resolves PyPI's newest, which is a 1.x
            reaches_one_x.append((job, slug, "unpinned"))
            continue
        try:
            specifier = Requirement(spec).specifier
        except InvalidRequirement:
            continue
        pins = [Version(str(s.version)) for s in specifier if s.operator == "=="]
        if any(v.major >= 1 and window.contains(str(v)) for v in pins):
            reaches_one_x.append((job, slug, spec))
    assert reaches_one_x, (
        f"no blocking lane in version-compat-ci.yml installs a TRL from the 1.x major "
        f"the window {window} now admits. Blocking lanes install: "
        f"{sorted({(j, s, p) for j, s, p in lanes})}. Lifting the cap without one means "
        f"the only lane that reaches 1.x is a continue-on-error canary, so a TRL 1.x "
        f"regression cannot fail this workflow."
    )


def test_the_checker_rejects_the_window_that_stranded_the_patches() -> None:
    """Negative control: every assertion above is a "nothing found" shape, which is also
    what a checker that has stopped checking reports."""
    shipped = SpecifierSet(">=0.18.2,!=0.19.0,<=0.24.0")
    assert "0.24.0" in shipped
    assert "1.13.0" not in shipped, "the old window must not admit the newly tested ceiling"
    for rejected in REJECTED:
        assert rejected not in shipped
    assert not shipped.contains("0.26.0")
    assert not shipped.contains("0.28.0")


def test_this_file_is_triggered_by_everything_it_scans() -> None:
    """A gate its own workflow's `paths:` filter cannot start is not a gate: the sweeps
    above read every `.github/workflows/*.yml` and `unsloth/models/rl_replacements.py`,
    so a PR that moves a cap in one of them and nothing else has to trigger this
    workflow."""
    if sys.version_info < (3, 11):
        pytest.skip("yaml parsing here needs the 3.11+ interpreter the job uses")
    import yaml

    workflow = yaml.safe_load((WORKFLOWS / "version-compat-ci.yml").read_text(encoding = "utf-8"))
    # PyYAML resolves a bare `on:` key to the boolean True (YAML 1.1), so read both.
    triggers = workflow.get("on", workflow.get(True)) or {}
    paths = (triggers.get("pull_request") or {}).get("paths") or []

    required = (".github/workflows/**", "unsloth/**", "tests/test_trl_cap_sites.py")
    missing = [name for name in required if name not in paths]
    assert not missing, (
        f"version-compat-ci.yml's pull_request.paths does not cover {missing}, so a PR "
        f"touching only those files never starts the cap-site-consistency job that reads "
        f"them. Current filter: {paths}"
    )


def test_a_stale_range_cap_is_caught_even_in_an_allowlisted_workflow(tmp_path, monkeypatch) -> None:
    """NEGATIVE CONTROL for the exemption: keyed on the filename, one intentional pin
    would exempt every other trl requirement in that file, so a range cap could go stale
    in the very workflow this gate scans."""
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "consolidated-tests-ci.yml").write_text(
        "run: |\n"
        "  pip install 'trl==0.18.2'\n"  # exact pin: a point, not a cap
        "  pip install 'trl>=0.18.2,<1.0.0'\n"  # the allowlisted half of the split
        "  pip install 'trl>=0.22,<0.26'\n",  # stale range cap: must be caught
        encoding = "utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "WORKFLOWS", workflows)

    with pytest.raises(AssertionError) as raised:
        test_no_workflow_lane_sits_below_the_declared_ceiling()

    message = str(raised.value)
    assert "consolidated-tests-ci.yml" in message
    assert "<0.26" in message
    assert "==0.18.2" not in message, "an exact pin is a point in the range, not a cap"
    assert "<1.0.0" not in message, "the allowlisted lane must stay allowlisted"


def test_an_unreachable_guard_is_caught(tmp_path, monkeypatch) -> None:
    """NEGATIVE CONTROL for the guard sweep: it must actually fail on a floor above the
    cap, not merely find nothing to complain about."""
    fake = tmp_path / "rl_replacements.py"
    fake.write_text(
        "def later():\n"
        '    if Version(importlib_version("trl")) < Version("99.0.0"):\n'
        "        return\n",
        encoding = "utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "RL_REPLACEMENTS", fake)

    with pytest.raises(AssertionError) as raised:
        test_no_trl_runtime_guard_is_unreachable_through_the_declared_window()

    assert "99.0.0" in str(raised.value)


def _steps_exposed_to_the_powershell_default(workflows: Path) -> list[tuple[str, str, str]]:
    """Every `run:` step a Windows runner hands to PowerShell, GitHub's default there. A
    step escapes only via `shell:` on itself, its job or the workflow; steps an `if:`
    gates off Windows are not exposed."""
    import yaml

    exposed = []
    for path in sorted(workflows.glob("*.yml")):
        try:
            document = yaml.safe_load(path.read_text(encoding = "utf-8")) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(document, dict):
            continue
        workflow_shell = ((document.get("defaults") or {}).get("run") or {}).get("shell")
        for job_name, job in (document.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            targets = f"{job.get('runs-on', '')}{job.get('strategy', '')}"
            if "windows" not in targets.lower():
                continue
            job_shell = ((job.get("defaults") or {}).get("run") or {}).get("shell")
            for index, step in enumerate(job.get("steps") or []):
                if not isinstance(step, dict):
                    continue
                body = step.get("run")
                if not isinstance(body, str):
                    continue
                if step.get("shell") or job_shell or workflow_shell:
                    continue
                condition = str(step.get("if", ""))
                if "ubuntu" in condition or "macos" in condition or "darwin" in condition:
                    continue
                name = step.get("name", f"step {index}")
                exposed.append((path.name, job_name, name, body))
    return exposed


def test_no_windows_step_uses_a_bash_line_continuation() -> None:
    r"""A trailing `\` is a bash continuation and NOT a PowerShell one.

    `cap-site-consistency` sets no `shell:`, so its windows-latest leg runs under
    PowerShell, which passes the `\` through as a literal argument. pytest read it as the
    path `\`, collected the whole working drive and died with `PermissionError:
    [WinError 5] ... 'D:\System Volume Information'` and 162 collection errors, having
    asserted nothing. Linux and macOS split it correctly, so the gate looked healthy while
    reporting two of its three runners.
    """
    if sys.version_info < (3, 11):
        pytest.skip("yaml parsing here needs the 3.11+ interpreter the job uses")

    offenders = [
        f"{workflow}: job {job!r} step {name!r}"
        for workflow, job, name, body in _steps_exposed_to_the_powershell_default(WORKFLOWS)
        if re.search(r"\\\s*\n", body)
    ]
    assert not offenders, (
        "these steps run under PowerShell on a Windows runner and use a bash `\\` line "
        "continuation, so the Windows leg silently runs a different command than the "
        "Linux one: " + "; ".join(offenders) + ". Put the command on one line or set "
        "`shell: bash` on the step."
    )


def test_the_powershell_continuation_check_can_fail(tmp_path, monkeypatch) -> None:
    """NEGATIVE CONTROL: proves the check finds the construct that broke the Windows leg."""
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "example-ci.yml").write_text(
        "jobs:\n"
        "  gate:\n"
        "    runs-on: windows-latest\n"
        "    steps:\n"
        "      - name: Assert every site declares the same window\n"
        "        run: |\n"
        "          python -m pytest tests/a.py \\\n"
        "            tests/b.py -v\n",
        encoding = "utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "WORKFLOWS", workflows)

    with pytest.raises(AssertionError) as raised:
        test_no_windows_step_uses_a_bash_line_continuation()
    assert "example-ci.yml" in str(raised.value)

    # ... and does NOT fire once the step declares a shell that understands `\`.
    (workflows / "example-ci.yml").write_text(
        "jobs:\n"
        "  gate:\n"
        "    runs-on: windows-latest\n"
        "    steps:\n"
        "      - name: Assert every site declares the same window\n"
        "        shell: bash\n"
        "        run: |\n"
        "          python -m pytest tests/a.py \\\n"
        "            tests/b.py -v\n",
        encoding = "utf-8",
    )
    test_no_windows_step_uses_a_bash_line_continuation()


def _ceiling_lane_trl_pins(workflows: Path) -> list[tuple[str, str, str]]:
    """(workflow, job, pinned trl version) per matrix lane with `slug: ceiling`; the pin
    is scanned out of the lane's single `pkg_pins` string."""
    import yaml

    found = []
    for path in sorted(workflows.glob("*.yml")):
        try:
            document = yaml.safe_load(path.read_text(encoding = "utf-8")) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(document, dict):
            continue
        for job_name, job in (document.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            # `strategy:` and `matrix:` can each be a bare `${{ }}` string, not a mapping.
            strategy = job.get("strategy")
            matrix = strategy.get("matrix") if isinstance(strategy, dict) else None
            includes = matrix.get("include") if isinstance(matrix, dict) else None
            for entry in includes or []:
                if not isinstance(entry, dict) or entry.get("slug") != "ceiling":
                    continue
                pins = str(entry.get("pkg_pins") or "")
                for match in re.finditer(r"trl==([0-9][0-9A-Za-z.\-]*)", pins):
                    found.append((path.name, job_name, match.group(1)))
    return found


def test_a_ceiling_lane_pins_the_declared_ceiling() -> None:
    """The assertion the range sweep structurally cannot make.

    That sweep exempts exact `==` pins, since a pin is a point not a cap and the floor
    lane's `trl==0.18.2` is correct. The shipped tree fell in the resulting hole: the
    `ceiling` lane pinned `transformers==5.17.0` next to `trl==0.24.0`, the post-lift
    transformers ceiling beside the pre-lift trl one. Nothing went red because 0.24.0 is
    still inside the window, just no longer its top. Scoped to `slug: ceiling`; floor and
    latest lanes pin what they like.
    """
    if sys.version_info < (3, 11):
        pytest.skip("yaml parsing here needs the 3.11+ interpreter the job uses")

    declared = _ceiling(_declared_window())
    lanes = _ceiling_lane_trl_pins(WORKFLOWS)
    assert lanes, (
        "no lane with `slug: ceiling` pins trl at all, so nothing measures the top of the "
        "declared window. If the ceiling lane was renamed, rename it here too."
    )
    stale = [
        f"{workflow}: job {job!r} pins trl=={pinned}"
        for workflow, job, pinned in lanes
        if Version(pinned) != declared
    ]
    assert not stale, (
        f"the declared trl ceiling is {declared}, but " + "; ".join(stale) + ". A lane "
        f"called `ceiling` that pins something else measures a range that is no longer the "
        f"edge of the window. Move the pin with the cap."
    )


def test_the_ceiling_lane_check_can_fail(tmp_path, monkeypatch) -> None:
    """NEGATIVE CONTROL: the exact drift that shipped, plus proof the check is not just
    asserting on its own input."""
    workflows = tmp_path / "workflows"
    workflows.mkdir()

    def write(trl_pin: str) -> None:
        (workflows / "version-compat-ci.yml").write_text(
            "jobs:\n"
            "  zoo-imports-under-spoof:\n"
            "    strategy:\n"
            "      matrix:\n"
            "        include:\n"
            "          - slug: floor\n"
            "            pkg_pins: \"'transformers==4.52.4' 'trl==0.18.2'\"\n"
            "          - slug: ceiling\n"
            f"            pkg_pins: \"'transformers==5.17.0' '{trl_pin}'\"\n",
            encoding = "utf-8",
        )

    monkeypatch.setattr(sys.modules[__name__], "WORKFLOWS", workflows)

    write("trl==0.24.0")
    with pytest.raises(AssertionError) as raised:
        test_a_ceiling_lane_pins_the_declared_ceiling()
    assert "0.24.0" in str(raised.value)

    # The floor lane's own exact pin must NOT be mistaken for a stale ceiling.
    write(f"trl=={TESTED_CEILING}")
    test_a_ceiling_lane_pins_the_declared_ceiling()
