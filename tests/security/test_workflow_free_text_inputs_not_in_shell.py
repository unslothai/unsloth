# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Free-text workflow inputs must reach `run:` scripts through env, never `${{ }}` (CWE-78)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

_EXPR = re.compile(r"\$\{\{(.*?)\}\}", re.S)
# Typed inputs (choice / boolean / number) cannot carry shell syntax; strings can.
_SAFE_TYPES = {"choice", "boolean", "number", "environment"}
# Event fields an actor controls as free text.
_EVENT_TEXT = re.compile(
    r"github\.(head_ref|event\.(pull_request\.(title|body|head\.ref|head\.label)"
    r"|issue\.(title|body)|comment\.body|review\.body|label\.name|client_payload))"
)


def _free_text_inputs(doc: dict) -> set[str]:
    on = doc.get("on", doc.get(True)) or {}
    if not isinstance(on, dict):
        return set()
    names = set()
    for trigger in ("workflow_dispatch", "workflow_call"):
        spec = on.get(trigger) or {}
        for name, meta in ((spec or {}).get("inputs") or {}).items():
            if (meta or {}).get("type", "string") not in _SAFE_TYPES:
                names.add(name)
    return names


def _offenders(path: Path) -> list[str]:
    doc = yaml.safe_load(path.read_text(encoding = "utf-8")) or {}
    free = _free_text_inputs(doc)
    found = []
    for job_name, job in (doc.get("jobs") or {}).items():
        for step in (job or {}).get("steps") or []:
            script = (step or {}).get("run")
            if not isinstance(script, str):
                continue
            for m in _EXPR.finditer(script):
                expr = m.group(1)
                refs = set(re.findall(r"\b(?:github\.event\.)?inputs\.([A-Za-z0-9_-]+)", expr))
                if refs & free or _EVENT_TEXT.search(expr):
                    found.append(f"{path.name}:{job_name}:{step.get('name')}: {m.group(0)}")
    return found


@pytest.mark.parametrize("path", sorted(WORKFLOWS.glob("*.y*ml")), ids = lambda p: p.name)
def test_no_free_text_expression_in_run(path):
    assert _offenders(path) == []


def test_detector_catches_interpolated_input(tmp_path):
    wf = tmp_path / "bad.yml"
    wf.write_text(
        "on:\n  workflow_dispatch:\n    inputs:\n      x:\n        type: string\n"
        "jobs:\n  j:\n    runs-on: ubuntu-latest\n    steps:\n"
        "      - run: echo '${{ inputs.x }}'\n"
        '      - env:\n          X: ${{ inputs.x }}\n        run: echo "$X"\n'
    )
    assert len(_offenders(wf)) == 1
