# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Studio smoke workflows trigger on what they can observe, and nothing else.

Every Studio smoke in this repo installs with `--local --no-torch`: the venv under test
cannot import the training library, so an edit under `unsloth/**` cannot change what the
job sees. Seven of the nine smokes nonetheless listed `unsloth/**` in their `pull_request`
filter, and all nine listed `studio/**`, which also matches `studio/src-tauri` (never built
by a smoke) and `studio/docs`. Measured over the last 20 PRs, a push that touched only
`studio/backend` fired 19 to 25 workflows and 50 to 90 jobs against an account that runs
about 30 to 35 jobs at once, and a training-only PR paid every Studio runner, macOS
included, for nothing. The queue waits that produced (p50 128 minutes on ubuntu-latest,
213 on macos-15) were the whole cost of the CI, the tests themselves finish in minutes.

Three rules, each cheap to break silently and expensive to leave broken:

1. No Studio smoke lists `unsloth/**` or a bare `studio/**`. The narrow set is spelled out
   per workflow (`studio/backend/**`, `studio/frontend/**` where Playwright drives the
   bundle, the installer entry points) and the comment above each list says why.
2. A filter lists every `.github/scripts` and `.github/actions` path the workflow executes,
   and nothing under those directories that it does not. The second half is what retired
   the `frontend-dist-*` entries from six workflows that never `uses:` them: a listed
   action the job never runs is a trigger for nothing, and it hid the fact that the list
   was copied rather than derived.
3. `local-agent-guides-ci.yml` names the route modules that serve the endpoints its
   preflight curls, not `studio/backend/routes/**`. Its connection matrix is nine
   ubuntu-latest cells of up to 16 minutes, and it fired for every edit to the training,
   export and dataset routes it never requests.

`tests/studio/test_macos_slots_per_commit.py` already pins `push.paths == pull_request.paths`
for the macOS workflows; this module does not repeat that.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

# Every workflow that installs Studio with --no-torch and boots it. The Update smokes are
# included because their filters were already narrow and must stay that way.
STUDIO_SMOKES = (
    "studio-api-smoke.yml",
    "studio-inference-smoke.yml",
    "studio-ui-smoke.yml",
    "studio-update-smoke.yml",
    "studio-windows-api-smoke.yml",
    "studio-windows-inference-smoke.yml",
    "studio-windows-ui-smoke.yml",
    "studio-windows-update-smoke.yml",
    "studio-mac-ui-smoke.yml",
)

AGENT_GUIDES = "local-agent-guides-ci.yml"

# Rule 2 applies to these on top of the smokes: they have the same shape of filter.
DERIVED_FILTERS = STUDIO_SMOKES + (AGENT_GUIDES,)

FORBIDDEN = {"unsloth/**", "studio/**"}

# A path under .github/scripts or .github/actions that a step runs, sources or `uses:`.
# `./.github/actions/x` and `.github/actions/x/action.yml` normalise to the same thing.
EXECUTED = re.compile(r"(?:\./)?(\.github/(?:scripts|actions)/[A-Za-z0-9_./-]+)")


def _on(doc):
    """The `on:` mapping, which PyYAML parses as the boolean True."""
    return doc.get(True) if True in doc else doc.get("on")


def _load(name: str):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding = "utf-8"))


def _paths(doc, event: str) -> list[str]:
    trigger = _on(doc).get(event)
    if not isinstance(trigger, dict):
        return []
    return list(trigger.get("paths") or [])


def _step_text(doc) -> str:
    """Everything a step can execute: run bodies, `uses:` targets, and `with:` values."""
    chunks = []
    for job in doc["jobs"].values():
        for step in job.get("steps") or []:
            for key in ("run", "uses"):
                if step.get(key):
                    chunks.append(str(step[key]))
            for value in (step.get("with") or {}).values():
                chunks.append(str(value))
            for value in (step.get("env") or {}).values():
                chunks.append(str(value))
    return "\n".join(chunks)


def _normalise(path: str) -> str:
    path = path.removeprefix("./")
    path = path.rstrip(".,;:)'\"")
    if path.startswith(".github/actions/") and not path.endswith(".yml"):
        path = path.rstrip("/") + "/action.yml"
    return path


# A helper script reaching a sibling: `$SCRIPT_DIR/x.sh`, `"$(dirname "$0")/x.txt"`, or
# the repo-relative `.github/scripts/x.sh` form.
SIBLING = re.compile(
    r"(?:\$SCRIPT_DIR|\$\{SCRIPT_DIR\}|\$\(dirname \"?\$0\"?\)|\.github/scripts)/([A-Za-z0-9_.-]+)"
)


def _executed_github_paths(doc) -> set[str]:
    """The .github/scripts and .github/actions files the steps reach.

    Indirection is followed: run-studio-ui-lane.sh boots Studio and drives the
    permission and indicator browsers through sibling scripts, agent-guides-drive.sh
    reads its prompts from sibling text files, and install-unsloth-local `uses:` the
    dist and uv cache actions. An edit to any of those changes what the job runs just
    as surely as an edit to the script or action the step names, so the filter has to
    list them too.
    """
    found = set()
    for match in EXECUTED.findall(_step_text(doc)):
        path = _normalise(match)
        if (REPO / path).is_file():
            found.add(path)
    # A local composite action that `uses:` another local action pulls that one in.
    pending = [p for p in found if p.startswith(".github/actions/")]
    while pending:
        text = (REPO / pending.pop()).read_text(encoding = "utf-8", errors = "replace")
        for match in EXECUTED.findall(text):
            nested = _normalise(match)
            if (
                nested.startswith(".github/actions/")
                and (REPO / nested).is_file()
                and nested not in found
            ):
                found.add(nested)
                pending.append(nested)
    for path in sorted(found):
        if not path.startswith(".github/scripts/"):
            continue
        text = (REPO / path).read_text(encoding = "utf-8", errors = "replace")
        for name in SIBLING.findall(text):
            sibling = f".github/scripts/{name}"
            if (REPO / sibling).is_file():
                found.add(sibling)
    return found


def _listed_github_paths(paths: list[str]) -> set[str]:
    return {
        p for p in paths if p.startswith(".github/scripts/") or p.startswith(".github/actions/")
    }


@pytest.mark.parametrize("name", STUDIO_SMOKES)
def test_no_studio_smoke_triggers_on_the_training_library_or_all_of_studio(name):
    doc = _load(name)
    for event in ("pull_request", "push"):
        offending = FORBIDDEN & set(_paths(doc, event))
        assert not offending, (
            f"{name} {event}.paths lists {sorted(offending)}. The install is --no-torch, so "
            "unsloth/** cannot reach the venv under test, and studio/** also matches the "
            "Tauri shell and docs the smoke never touches. Name the directories the job "
            "observes instead."
        )
    assert "studio/backend/**" in _paths(doc, "pull_request") or name.endswith(
        "update-smoke.yml"
    ), f"{name} must still trigger on the backend it boots"


@pytest.mark.parametrize("name", STUDIO_SMOKES)
def test_every_smoke_still_names_the_workflow_file_and_the_apt_helper_it_runs(name):
    doc = _load(name)
    paths = set(_paths(doc, "pull_request"))
    assert f".github/workflows/{name}" in paths, f"{name} must re-run when it is edited"
    text = _step_text(doc)
    if "retry-with-apt-lock.sh" in text:
        assert (
            ".github/scripts/retry-with-apt-lock.sh" in paths
        ), f"{name} calls retry-with-apt-lock.sh but would not re-run when it changes"


@pytest.mark.parametrize("name", DERIVED_FILTERS)
def test_the_filter_lists_exactly_the_github_paths_the_steps_execute(name):
    doc = _load(name)
    executed = _executed_github_paths(doc)
    for event in ("pull_request", "push"):
        paths = _paths(doc, event)
        if not paths:
            continue
        listed = _listed_github_paths(paths) - {f".github/workflows/{name}"}
        missing = sorted(executed - listed)
        assert not missing, (
            f"{name} {event}.paths does not list {missing}, which its steps execute; an edit "
            "to any of them changes what the job runs without running it"
        )
        dead = sorted(listed - executed)
        assert not dead, (
            f"{name} {event}.paths lists {dead}, which no step in the workflow runs, sources "
            "or `uses:`; a trigger for nothing means the list was copied, not derived"
        )


def test_executed_path_detection_is_not_vacuous():
    """The scanner must find the helpers every smoke is known to call."""
    doc = _load("studio-ui-smoke.yml")
    executed = _executed_github_paths(doc)
    assert ".github/scripts/boot-studio-api-only.sh" in executed
    assert ".github/actions/install-unsloth-local/action.yml" in executed
    # Reached through install-unsloth-local, which `uses:` the dist and uv cache pairs.
    assert ".github/actions/frontend-dist-restore/action.yml" in executed
    assert ".github/actions/uv-cache-restore/action.yml" in executed
    windows = _executed_github_paths(_load("studio-windows-ui-smoke.yml"))
    assert ".github/actions/frontend-dist-restore/action.yml" in windows


def _endpoints_curled(doc) -> set[str]:
    """The /v1 endpoints the workflow's own steps request."""
    return set(re.findall(r"/v1/(chat/completions|messages|responses|models)\b", _step_text(doc)))


def _route_decorators(module: Path) -> set[str]:
    """The route suffixes a module declares, e.g. chat/completions for /v1/chat/completions."""
    text = module.read_text(encoding = "utf-8")
    found = set()
    for path in re.findall(r"@router\.(?:get|post|api_route)\(\s*\"/?([A-Za-z0-9_/{}:]+)\"", text):
        found.add(path.strip("/").removeprefix("v1/"))
    return found


def test_agent_guides_lists_the_route_modules_that_serve_what_it_curls():
    doc = _load(AGENT_GUIDES)
    curled = _endpoints_curled(doc)
    assert {"chat/completions", "messages", "responses", "models"} <= curled, curled
    for event in ("pull_request", "push"):
        paths = _paths(doc, event)
        assert "studio/backend/routes/**" not in paths, (
            f"{AGENT_GUIDES} {event}.paths matches every route module; the preflight only "
            "requests the OpenAI and Anthropic surfaces"
        )
        routes = sorted(p for p in paths if p.startswith("studio/backend/routes/"))
        assert routes, f"{AGENT_GUIDES} {event}.paths names no route module at all"
        serving = set()
        for entry in routes:
            module = REPO / entry
            assert module.is_file(), f"{AGENT_GUIDES} lists {entry}, which does not exist"
            declared = _route_decorators(module)
            served = {e for e in curled if e in declared or e.rstrip("/") + "/" in declared}
            assert served, (
                f"{AGENT_GUIDES} lists {entry} but it declares none of {sorted(curled)}; "
                "either the endpoint moved or the entry is stale"
            )
            serving |= served
        assert curled <= serving, (
            f"{AGENT_GUIDES} {event}.paths covers {sorted(serving)} but the preflight also "
            f"requests {sorted(curled - serving)}; add the module that serves it"
        )
        # The contracts outside routes/ the docstring names must stay.
        for required in (
            "studio/backend/main.py",
            "studio/backend/core/inference/llama_cpp.py",
            "studio/backend/models/**",
            "unsloth_cli/*",
            "unsloth_cli/commands/*.py",
        ):
            assert required in paths, f"{AGENT_GUIDES} {event}.paths lost {required}"
        assert "unsloth_cli/**" not in paths, (
            f"{AGENT_GUIDES} {event}.paths matches the whole CLI package; the cells never run "
            "unsloth_cli/tests/**"
        )
