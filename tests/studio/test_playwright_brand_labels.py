# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Branded labels the Playwright drivers wait for still exist in the app.

#14473 renamed "Studio-managed folder" to "Unsloth-managed folder" in the working folder picker, and
playwright_project_folders.py kept waiting for the old text, so the browser smoke failed on every PR
that ran it. A rename is caught here, in seconds, instead of in a browser job.
"""

import ast
import functools
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DRIVERS = sorted((REPO / "tests" / "studio").glob("playwright_*.py"))
BRANDS = ("Unsloth", "Studio")
LOCATORS = {"get_by_text", "get_by_label", "get_by_placeholder", "get_by_title"}


@functools.cache
def _app_text():
    out = subprocess.run(
        [
            "git",
            "grep",
            "-h",
            "-I",
            "-e",
            "Unsloth",
            "-e",
            "Studio",
            "--",
            "studio/frontend/src",
            "studio/backend",
        ],
        cwd = REPO,
        capture_output = True,
        text = True,
        check = True,
    )
    return out.stdout


def _constants(tree):
    names = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                    names[target.id] = node.value.value
    return names


def _branded_labels(path):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    names = _constants(tree)

    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return names.get(node.id)
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        values = []
        if node.func.attr in LOCATORS and node.args:
            values.append(literal(node.args[0]))
        if node.func.attr == "get_by_role":
            values += [literal(k.value) for k in node.keywords if k.arg == "name"]
        for value in values:
            if value and any(brand in value for brand in BRANDS):
                yield node.lineno, value


def test_the_scan_sees_the_drivers():
    assert DRIVERS, "no tests/studio/playwright_*.py drivers found; update this guard"
    found = [label for path in DRIVERS for _, label in _branded_labels(path)]
    assert "Unsloth-managed folder" in found


@pytest.mark.parametrize("driver", DRIVERS, ids = lambda p: p.name)
def test_branded_labels_exist_in_the_app(driver):
    app = _app_text()
    missing = [
        f"{driver.name}:{line} {label!r}"
        for line, label in _branded_labels(driver)
        if label not in app
    ]
    assert not missing, "Playwright waits for text the app no longer renders: " + ", ".join(missing)
