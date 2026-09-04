# Copyright 2026-present the Unforgettable contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolve a nominated test command without importing Studio."""

from __future__ import annotations

import json
from pathlib import Path

from unforgettable.store.records import list_records
from unforgettable.store.titles import normalize_title

TEST_COMMAND_TITLE = "test command"


def first_nonempty_line(body: str) -> str:
    for line in (body or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def resolve_test_command(
    *,
    requested: str | None,
    db_path = None,
    tree: Path | None = None,
) -> str | None:
    if requested and requested.strip():
        return requested.strip()
    for rec in list_records(kinds = ["procedure"], statuses = ["active"], db_path = db_path):
        if normalize_title(rec["title"]) == TEST_COMMAND_TITLE:
            cmd = first_nonempty_line(rec["body"])
            if cmd:
                return cmd
            break
    if tree is not None:
        return detect_test_command(tree)
    return None


def detect_test_command(tree: Path) -> str | None:
    try:
        root = Path(tree)
        if not root.is_dir():
            return None
    except OSError:
        return None

    def _read(path: Path) -> str | None:
        try:
            if not path.is_file():
                return None
            return path.read_text(encoding = "utf-8", errors = "replace")
        except OSError:
            return None

    if _read(root / "pytest.ini") is not None:
        return "pytest"
    pyproject = _read(root / "pyproject.toml")
    if pyproject is not None and "[tool.pytest" in pyproject:
        return "pytest"
    package_text = _read(root / "package.json")
    if package_text is not None:
        try:
            data = json.loads(package_text)
        except json.JSONDecodeError:
            data = None
        if (
            isinstance(data, dict)
            and isinstance(data.get("scripts"), dict)
            and data["scripts"].get("test")
        ):
            return "npm test"
    if _read(root / "go.mod") is not None:
        return "go test ./..."
    return None
