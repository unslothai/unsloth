# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio smoke filters list only paths they can observe; no unsloth/** and no bare studio/**."""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

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

DERIVED_FILTERS = STUDIO_SMOKES + (AGENT_GUIDES,)

FORBIDDEN = {"unsloth/**", "studio/**"}

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


SIBLING = re.compile(
    r"(?:\$SCRIPT_DIR|\$\{SCRIPT_DIR\}|\$\(dirname \"?\$0\"?\)|\.github/scripts)/([A-Za-z0-9_.-]+)"
)


def _executed_github_paths(doc) -> set[str]:
    """Follows indirection: a sibling script or prompt file the step reads changes what the job runs."""
    found = set()
    for match in EXECUTED.findall(_step_text(doc)):
        path = _normalise(match)
        if (REPO / path).is_file():
            found.add(path)
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
