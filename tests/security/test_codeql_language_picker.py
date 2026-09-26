# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The CodeQL workflow's language picker, run for real with a fake ``gh`` on PATH.

A pull request analyses only the languages its files touch, so a picker that misses a
language silently drops CodeQL coverage for that change. Every push, schedule and dispatch
must analyse all four.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "codeql.yml"
ALL = ["actions", "javascript-typescript", "python", "rust"]

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason = "needs bash"
)


def _script() -> str:
    doc = yaml.safe_load(WORKFLOW.read_text())
    return doc["jobs"]["changes"]["steps"][0]["run"]


def _pick(
    tmp_path: Path,
    files: list[str] | None,
    event: str = "pull_request",
) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok = True)
    listing = tmp_path / "files.txt"
    gh = bin_dir / "gh"
    if files is None:
        gh.write_text("#!/bin/bash\nexit 1\n")
    else:
        listing.write_text("".join(f + "\n" for f in files))
        gh.write_text(f"#!/bin/bash\ncat '{listing}'\n")
    gh.chmod(0o755)
    out = tmp_path / "out.txt"
    out.write_text("")
    env = dict(
        os.environ,
        PATH = f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        GITHUB_OUTPUT = str(out),
        EVENT = event,
        REPO = "unslothai/unsloth",
        PR = "1",
    )
    subprocess.run(["bash", "-c", _script()], env = env, check = True, capture_output = True)
    return json.loads(out.read_text().strip().split("=", 1)[1])


@pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
def test_non_pull_request_events_analyse_everything(tmp_path, event):
    assert _pick(tmp_path, ["README.md"], event) == ALL


@pytest.mark.parametrize(
    "files, expected",
    [
        (["unsloth/models/llama.py"], ["python"]),
        (["studio/backend/stubs.pyi"], ["python"]),
        (["studio/frontend/src/app.tsx"], ["javascript-typescript"]),
        (["studio/frontend/package.json"], ["javascript-typescript"]),
        (["studio/frontend/tsconfig.app.json"], ["javascript-typescript"]),
        (["studio/frontend/index.html"], ["javascript-typescript"]),
        (["studio/src-tauri/src/main.rs"], ["rust"]),
        (["studio/src-tauri/Cargo.lock"], ["rust"]),
        ([".github/workflows/lint-ci.yml"], ["actions"]),
        ([".github/actions/setup/action.yml"], ["actions"]),
        (["README.md", "docs/guide.md", "studio/frontend/public/logo.png"], []),
        (["a.py", "b.ts", "c.rs", ".github/workflows/x.yml"], ALL),
        # A rename lists both names; the old one alone still counts.
        (["docs/moved.txt", "unsloth/old_name.py"], ["python"]),
    ],
)
def test_pull_requests_analyse_the_languages_they_touch(tmp_path, files, expected):
    assert _pick(tmp_path, files) == expected


def test_a_codeql_config_change_analyses_everything(tmp_path):
    assert _pick(tmp_path, [".github/workflows/codeql.yml"]) == ALL
    assert _pick(tmp_path, [".github/codeql/codeql-config.yml"]) == ALL


def test_an_unreadable_file_list_analyses_everything(tmp_path):
    assert _pick(tmp_path, None) == ALL


def test_the_api_file_limit_analyses_everything(tmp_path):
    assert _pick(tmp_path, [f"docs/{i}.md" for i in range(3000)]) == ALL


def test_an_early_match_in_a_long_list_is_not_lost(tmp_path):
    """grep -q stops reading at its first match; fed from a pipe under pipefail, the
    writer's SIGPIPE used to turn that match into a miss."""
    files = ["first.py"] + [f"docs/{'x' * 200}/{i}.md" for i in range(2998)]
    assert _pick(tmp_path, files) == ["python"]
