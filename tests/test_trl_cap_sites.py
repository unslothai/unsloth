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

# The datasets ceiling every unsloth_zoo up to and including 2026.9.7 publishes, and the reason
# the widened datasets window below is advertised rather than delivered: pip intersects, so users
# keep resolving under 4.4.0 whatever this file says.
ZOO_DATASETS_CEILING_BEFORE_THE_LIFT = SpecifierSet("<4.4.0")

# DEFERRED, unlike the trl half: no published zoo carries the widened datasets window yet. A floor
# no release satisfies makes unsloth uninstallable, so this stays None and the gate below skips
# until test_the_zoo_datasets_deferral_expires_when_the_zoo_release_ships goes red on its own.
ZOO_FLOOR_WITH_LIFTED_DATASETS_CAP = None

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
    """The TIGHTEST upper bound: `max` reported 1.13.0 for `<=0.24.0,<=1.13.0`, which resolves at
    0.24.0, so the assertions passed on exactly the stale cap this file exists to catch.
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
    """pip gives the user the LOWER ceiling, and every zoo up to 2026.9.4 says `trl<=0.24.0`, under
    which the guards above stay unreachable. 2026.9.5 is the release carrying the lift.
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
    """A patch gated on a TRL newer than the cap admits is dead code that ships green: under
    `<=0.24.0` both `openenv_vllm_reload_weights` and `vllm_generation_init_patch` returned early.
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
    """Every (job, lane, trl spec) that can actually fail the run. Shape-agnostic: that lane has been
    an explicit pin and an unpinned `latest`, blocking and `continue-on-error`.
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
    """Satisfied by a blocking lane pinning an explicit 1.x or leaving trl unpinned, NOT by a
    `continue-on-error` one: a canary as the sole 1.x lane advertises a range no gate defends.
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
    """A gate its own workflow's `paths:` filter cannot start is not a gate: the sweeps read every
    workflow and rl_replacements.py, so moving a cap in one of them has to trigger this workflow.
    """
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
    """NEGATIVE CONTROL: keyed on the filename, one intentional pin would exempt every other trl
    requirement in that file, so a range cap could go stale in the very workflow this scans.
    """
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


# `linux` is here because a matrix leg is as often keyed on an artifact or label
# (`startsWith(matrix.artifact, 'linux-')`) as on the image name, and reading only `ubuntu`
# reported a Linux-only apt step as a PowerShell offender.
_NON_WINDOWS_TOKENS = ("ubuntu", "linux", "macos", "darwin", "mac-", "'mac'")


def _gated_off_windows(condition: str) -> bool:
    """Whether an `if:` keeps its step off the Windows leg. Naming a non-Windows OS excludes Windows
    only when POSITIVE: `matrix.os != 'ubuntu-latest'` is precisely the condition that RUNS there,
    so a negated condition stays reported as exposed.
    """
    lowered = condition.lower()
    if not any(token in lowered for token in _NON_WINDOWS_TOKENS):
        return False
    return "!=" not in lowered and "!" not in lowered


def _steps_exposed_to_the_powershell_default(workflows: Path) -> list[tuple[str, str, str]]:
    """Every `run:` step a Windows runner hands to PowerShell. A step escapes only via `shell:` on
    itself, its job or the workflow; an `if:` gating it off Windows is not exposed.
    """
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
                if _gated_off_windows(str(step.get("if", ""))):
                    continue
                name = step.get("name", f"step {index}")
                exposed.append((path.name, job_name, name, body))
    return exposed


def test_no_windows_step_uses_a_bash_line_continuation() -> None:
    r"""PowerShell passes a trailing `\` through as a literal argument: pytest read it as the path
    `\`, collected the whole drive and died with 162 collection errors having asserted nothing,
    while Linux and macOS split it correctly and the gate looked healthy.
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


@pytest.mark.parametrize(
    "condition,gated",
    [
        ("", False),
        ("startsWith(matrix.artifact, 'linux-')", True),
        ("runner.os == 'Linux'", True),
        ("matrix.os == 'ubuntu-latest'", True),
        ("runner.os == 'macOS'", True),
        # The negated forms run ON Windows, so naming a non-Windows OS must not clear them.
        ("matrix.os != 'ubuntu-latest'", False),
        ("!startsWith(matrix.artifact, 'linux-')", False),
        # Nothing about an OS at all: still exposed.
        ("github.event_name == 'pull_request'", False),
    ],
)
def test_only_a_positive_non_windows_guard_clears_a_step(condition: str, gated: bool) -> None:
    """The `!=` case is the one that matters: it is how a step lands on Windows."""
    assert _gated_off_windows(condition) is gated


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
    """The range sweep exempts exact `==` pins, and the shipped tree fell in that hole: the `ceiling`
    lane pinned `transformers==5.17.0` beside `trl==0.24.0`, still inside the window, no longer its top.
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


# The ceiling has to be RESOLVABLE, not merely declared: a sibling requirement narrows it too
# and pip backtracks silently rather than erroring. With `trl<=1.13.0` next to `datasets<4.4.0`
# pip walked back through all thirty 1.x releases and installed 0.29.1, since every trl 1.x
# needs `datasets>=4.7.0`, and nothing was red. Recorded rather than fetched so this runs in the
# dependency-free three-OS job; `test_the_recorded_trl_datasets_floors_still_match_pypi`
# re-derives it when the network is there.
TRL_DATASETS_FLOORS = (
    # (first trl release with this floor, the datasets floor it declares)
    (Version("0.18.2"), Version("3.0.0")),
    (Version("1.0.0"), Version("4.7.0")),
)


def _pyproject_requirement(name: str) -> SpecifierSet:
    """The single declared WINDOW for `name`. Exact `==` pins are skipped rather than counted as a
    second window: a pin says what the fully-pinned Studio single-env ships, and moving one needs
    its own lockfile, so counting them together would fire on a deliberate disagreement.
    """
    data = _toml()
    project = data.get("project") or {}
    raws: list[str] = list(project.get("dependencies") or [])
    for extra in (project.get("optional-dependencies") or {}).values():
        raws.extend(extra)
    windows = set()
    for raw in raws:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        if req.name.lower().replace("_", "-") != name:
            continue
        if all(spec.operator == "==" for spec in req.specifier) and len(req.specifier):
            continue
        windows.add(str(req.specifier))
    assert windows, f"pyproject.toml declares no {name} window at all, only exact pins"
    assert len(windows) == 1, (
        f"pyproject.toml declares {len(windows)} different {name} windows across its "
        f"extras: {sorted(windows)}. One of them will be the one users hit and the other "
        f"will be the one CI tests."
    )
    return SpecifierSet(windows.pop())


def _datasets_floor_for(trl_version: Version) -> Version:
    floor = TRL_DATASETS_FLOORS[0][1]
    for first, declared in TRL_DATASETS_FLOORS:
        if trl_version >= first:
            floor = declared
    return floor


def _datasets_releases() -> list[Version]:
    """Every datasets release the declared window could pick, recorded for the same reason."""
    return [
        Version(v)
        for v in (
            "3.4.1",
            "3.5.0",
            "3.6.0",
            "4.0.0",
            "4.1.0",
            "4.2.0",
            "4.3.0",
            "4.4.0",
            "4.4.1",
            "4.4.2",
            "4.5.0",
            "4.6.0",
            "4.6.1",
            "4.7.0",
            "4.8.0",
            "4.8.5",
            "5.0.0",
            "5.0.1",
        )
    ]


def test_the_declared_datasets_window_can_supply_the_declared_trl_ceiling() -> None:
    """The assertion the shipped cap failed: a ceiling no sibling window lets pip reach."""
    ceiling = _ceiling(_declared_window())
    needed = _datasets_floor_for(ceiling)
    datasets_window = _pyproject_requirement("datasets")
    usable = [v for v in _datasets_releases() if str(v) in datasets_window and v >= needed]
    assert usable, (
        f"the declared trl ceiling {ceiling} requires datasets>={needed}, but the declared "
        f"datasets window {datasets_window} admits no such release. pip does not error on "
        f"this: it backtracks to the newest trl whose datasets floor fits and installs that "
        f"instead, so the advertised ceiling is one no install can reach and nothing goes "
        f"red. Move the datasets window with the trl one, or lower the trl ceiling to the "
        f"newest release this datasets window can actually supply."
    )


def test_the_resolvability_check_can_fail() -> None:
    """NEGATIVE CONTROL: the window that shipped, and the one that fixes it."""
    stranded = SpecifierSet(">=3.4.1,!=4.0.*,!=4.1.0,<4.4.0")
    needed = _datasets_floor_for(Version("1.13.0"))
    assert not [v for v in _datasets_releases() if str(v) in stranded and v >= needed], (
        "datasets<4.4.0 must not be able to supply a trl 1.x: that pairing is exactly the "
        "silently-unreachable ceiling this check exists to catch."
    )
    opened = SpecifierSet(">=3.4.1,!=4.0.*,!=4.1.0,!=4.4.*,!=4.5.0,<5.0.0")
    assert [v for v in _datasets_releases() if str(v) in opened and v >= needed]


def _runtime_forbidden_datasets_range() -> tuple[Version, Version]:
    """The (low, high) datasets range `patch_datasets` refuses at import, read off the guard.

    Read rather than restated, so the range cannot be edited in one place only.
    """
    guard = RL_REPLACEMENTS.parent.parent / "import_fixes.py"
    source = guard.read_text(encoding = "utf-8")
    match = re.search(
        r'datasets_version <= Version\("([\d.]+)"\).*?datasets_version >= Version\("([\d.]+)"\)',
        source,
        re.S,
    )
    assert match, "patch_datasets no longer states its forbidden range in the shape this reads"
    return Version(match.group(2)), Version(match.group(1))


def _workflow_datasets_specs(path: Path) -> list[tuple[str, Requirement]]:
    """Every `datasets<spec>` requirement spelled inside a workflow's shell steps."""
    text = path.read_text(encoding = "utf-8")
    out = []
    for match in re.finditer(r"""['"](datasets[<>=!,.*\d\s]*)['"]""", text):
        try:
            req = Requirement(match.group(1))
        except InvalidRequirement:
            continue
        if req.name.lower() == "datasets" and str(req.specifier):
            out.append((match.group(1), req))
    return out


def test_no_workflow_lane_installs_a_datasets_the_runtime_guard_refuses() -> None:
    """`notebooks-ci.yml` carried `datasets>=3.4,<5`, admitting 4.4.x and 4.5.0; that lane imports
    unsloth, so the job dies at import having tested nothing, green only because pip picked newer.
    """
    low, high = _runtime_forbidden_datasets_range()
    refused = [v for v in _datasets_releases() if low <= v <= high]
    assert refused, "no recorded datasets release falls in the guard's forbidden range"

    offenders: dict[str, list[str]] = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for raw, req in _workflow_datasets_specs(path):
            # Exact pins are NOT exempt, unlike the trl ceiling check: `datasets==4.4.0` chooses
            # a release that cannot import, not a point inside a supported range.
            admitted = [v for v in refused if str(v) in req.specifier]
            if admitted:
                offenders.setdefault(path.name, []).append(
                    f"{raw.strip()} admits {', '.join(str(v) for v in admitted)}"
                )
    assert not offenders, (
        f"these workflow lanes can resolve a datasets that patch_datasets refuses at import "
        f"[{low}, {high}], so the job dies on `import unsloth` rather than testing anything: "
        f"{offenders}. Spell the declared pyproject window instead."
    )


def test_the_workflow_datasets_check_can_fail(tmp_path, monkeypatch) -> None:
    """NEGATIVE CONTROL: the loose bound that shipped must be caught."""
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "example-ci.yml").write_text(
        "jobs:\n  gate:\n    steps:\n      - run: pip install 'datasets>=3.4,<5'\n",
        encoding = "utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "WORKFLOWS", workflows)
    with pytest.raises(AssertionError) as raised:
        test_no_workflow_lane_installs_a_datasets_the_runtime_guard_refuses()
    assert "example-ci.yml" in str(raised.value)

    # ... and the declared window is accepted.
    (workflows / "example-ci.yml").write_text(
        "jobs:\n  gate:\n    steps:\n      - run: pip install "
        "'datasets>=3.4.1,!=4.0.*,!=4.1.0,!=4.4.*,!=4.5.0,<5.0.0'\n",
        encoding = "utf-8",
    )
    test_no_workflow_lane_installs_a_datasets_the_runtime_guard_refuses()


def test_the_datasets_window_still_excludes_exactly_what_the_runtime_guard_refuses() -> None:
    """Not "at least as strict": a WIDER window installs a release the guard refuses at import, and a
    NARROWER one (the shipped `<4.4.0`) forbids releases nothing objects to, stranding the ceiling.
    """
    low, high = _runtime_forbidden_datasets_range()

    window = _pyproject_requirement("datasets")
    for release in _datasets_releases():
        refused_at_runtime = low <= release <= high
        admitted_by_metadata = str(release) in window
        if refused_at_runtime:
            assert not admitted_by_metadata, (
                f"datasets {release} is inside patch_datasets' forbidden range "
                f"[{low}, {high}] but the declared window still admits it, so pip may "
                f"install a release that raises NotImplementedError at import."
            )


def test_the_recorded_trl_datasets_floors_still_match_pypi() -> None:
    """Re-derive TRL_DATASETS_FLOORS from PyPI. Skips offline; this is the only network here."""
    import json
    import urllib.error
    import urllib.request

    def metadata(version: Version) -> list[str]:
        url = f"https://pypi.org/pypi/trl/{version}/json"
        try:
            with urllib.request.urlopen(url, timeout = 15) as response:
                return json.load(response)["info"].get("requires_dist") or []
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            pytest.skip(f"PyPI unreachable ({error})")

    for first, declared in TRL_DATASETS_FLOORS:
        floors = [
            Requirement(raw).specifier
            for raw in metadata(first)
            if Requirement(raw).name.lower() == "datasets" and not Requirement(raw).marker
        ]
        assert floors, f"trl {first} declares no unconditional datasets requirement"
        assert str(declared) in str(floors[0]), (
            f"trl {first} now declares datasets{floors[0]}, not >={declared}. Update "
            f"TRL_DATASETS_FLOORS, then re-check the declared datasets window against it."
        )


def _newest_published_zoo_datasets_windows(timeout: float = 10.0):
    """(zoo version, datasets windows) for the newest unsloth_zoo on PyPI, else None.

    Never raises: the caller treats "could not ask" as "keep deferring".
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

    windows = []
    for raw in info.get("requires_dist") or []:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        if req.name.lower() != "datasets" or not req.specifier:
            continue
        windows.append(req.specifier)
    if not windows:
        return None
    return Version(released), windows


def test_the_declared_zoo_floor_can_supply_the_declared_datasets_window() -> None:
    """The datasets twin of the trl gate: pip hands the user the LOWER of the two ceilings.

    Deferred while ZOO_FLOOR_WITH_LIFTED_DATASETS_CAP is None; the body is kept so naming the
    release later is one edit.
    """
    if ZOO_FLOOR_WITH_LIFTED_DATASETS_CAP is None:
        pytest.skip(
            "deferred: no published unsloth_zoo carries the widened datasets window, so pip "
            "keeps intersecting it back to the zoo's own cap. Re-enable by setting "
            "ZOO_FLOOR_WITH_LIFTED_DATASETS_CAP to the release that carries it."
        )
    reqs = _pyproject_zoo()
    assert reqs, "pyproject.toml names no versioned unsloth_zoo requirement at all"
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
        if floor < ZOO_FLOOR_WITH_LIFTED_DATASETS_CAP
    }
    assert not stale, (
        f"pyproject.toml widens datasets while still accepting unsloth_zoo {stale}, whose own "
        f"cap is {ZOO_DATASETS_CEILING_BEFORE_THE_LIFT}. Raise the floor to "
        f"{ZOO_FLOOR_WITH_LIFTED_DATASETS_CAP} in the same commit as the window."
    )


def test_the_zoo_datasets_deferral_expires_when_the_zoo_release_ships() -> None:
    """Self-expiring, like the transformers one: a deferral nothing can end is the defect.

    Asks PyPI what the newest zoo admits and fails only on POSITIVE evidence the deferral is
    obsolete, so no network, a timeout or an unreadable answer all leave it skipped.
    """
    if ZOO_FLOOR_WITH_LIFTED_DATASETS_CAP is not None:
        pytest.skip("the floor already names a zoo carrying the lift, so the gate above is live")

    published = _newest_published_zoo_datasets_windows()
    if published is None:
        pytest.skip("PyPI could not be asked for unsloth_zoo, so the deferral stands")

    zoo_version, zoo_windows = published
    # The newest datasets our own window admits that the zoo's does not. Membership, not a number
    # comparison, for the reason the transformers twin gives: `<4.4.0` and `<=4.4.0` differ.
    declared = _pyproject_requirement("datasets")
    probe = [v for v in _datasets_releases() if str(v) in declared]
    assert probe, "the declared datasets window admits no recorded release at all"
    newest = max(probe)
    covering = [str(w) for w in zoo_windows if w.contains(str(newest))]
    assert not covering, (
        f"unsloth_zoo {zoo_version} is published and admits datasets {newest} "
        f"({', '.join(covering)}), so the deferral is over. Set "
        f"ZOO_FLOOR_WITH_LIFTED_DATASETS_CAP to {zoo_version} and raise the unsloth_zoo floor "
        f"in pyproject.toml to it in the same commit; that re-enables "
        f"test_the_declared_zoo_floor_can_supply_the_declared_datasets_window."
    )


def test_the_zoo_datasets_expiry_can_fail(monkeypatch) -> None:
    """NEGATIVE CONTROL: the expiry is a "nothing found" shape, which is also what a check that
    has stopped checking reports. It must fire on a zoo that covers the window, and stay silent
    when PyPI cannot be asked, or an offline three-OS runner turns red for the wrong reason.
    """
    module = sys.modules[__name__]

    monkeypatch.setattr(
        module,
        "_newest_published_zoo_datasets_windows",
        lambda timeout = 10.0: (Version("2026.9.9"), [SpecifierSet(">=3.4.1,<5.0.0")]),
    )
    with pytest.raises(AssertionError) as raised:
        test_the_zoo_datasets_deferral_expires_when_the_zoo_release_ships()
    assert "2026.9.9" in str(raised.value)

    monkeypatch.setattr(module, "_newest_published_zoo_datasets_windows", lambda timeout = 10.0: None)
    with pytest.raises(pytest.skip.Exception):
        test_the_zoo_datasets_deferral_expires_when_the_zoo_release_ships()
