# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/codex-agent-workspace-ci.yml"
ON = True  # PyYAML reads the unquoted YAML `on` key as boolean true.


def _workflow() -> dict:
    document = yaml.safe_load(WORKFLOW_PATH.read_text(encoding = "utf-8"))
    assert isinstance(document, dict)
    return document


def _step(job: dict, name: str) -> dict:
    for step in job.get("steps", []):
        if step.get("name") == name:
            return step
    raise AssertionError(f"missing workflow step: {name}")


def test_workspace_contracts_run_on_all_supported_platforms() -> None:
    job = _workflow()["jobs"]["workspace-contracts"]
    assert job["strategy"]["matrix"]["os"] == ["ubuntu-latest", "macos-15", "windows-latest"]


def test_durable_research_lane_covers_handoff_progress_and_storage() -> None:
    job = _workflow()["jobs"]["workspace-contracts"]
    command = _step(job, "Run durable research workspace contracts")["run"]
    for test_file in (
        "studio/backend/tests/test_deep_research_handoff_simulation.py",
        "studio/backend/tests/test_research_progress_events.py",
        "studio/backend/tests/test_research_runs_storage.py",
    ):
        assert test_file in command


def test_frontend_lane_runs_the_full_contract_suite() -> None:
    job = _workflow()["jobs"]["workspace-contracts"]
    command = _step(job, "Run full frontend contract suite")["run"]
    assert command.strip() == "npm test"


def test_native_lane_runs_full_tests_on_all_supported_platforms() -> None:
    job = _workflow()["jobs"]["native-contracts"]
    assert job["strategy"]["matrix"]["os"] == ["ubuntu-22.04", "macos-15", "windows-latest"]
    command = _step(job, "Run full Tauri contract suite")["run"]
    assert command.strip() == "cargo test -- --test-threads=1"


def test_matrix_guard_and_research_contracts_trigger_the_workflow() -> None:
    triggers = _workflow()[ON]
    required_paths = {
        "studio/backend/tests/test_deep_research_handoff_simulation.py",
        "studio/backend/tests/test_research_progress_events.py",
        "studio/backend/tests/test_research_runs_storage.py",
        "tests/studio/test_codex_agent_workspace_ci_matrix.py",
    }
    for event in ("pull_request", "push"):
        assert required_paths <= set(triggers[event]["paths"])
