# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""studio-windows-installer-smoke.yml runs only on what small-checks reads, so every file the
Pester suite opens and every module the prebuilt resolver imports must match its path filters."""

import ast
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "studio-windows-installer-smoke.yml"
PESTER = REPO / "tests" / "studio_setup_ps1"
STUDIO = REPO / "studio"

# Join-Path $PSScriptRoot '..\..\install.ps1'
_FROM_SCRIPT_ROOT = re.compile(r"""Join-Path\s+\$PSScriptRoot\s+['"]([^'"]+)['"]""")
# Join-Path (Split-Path -Parent $script:SetupPs1) 'install_node_prebuilt.py'
_FROM_SETUP_DIR = re.compile(
    r"""Join-Path\s+\(Split-Path\s+-Parent\s+\$script:SetupPs1\)\s+['"]([^'"]+)['"]"""
)


def _glob(pattern: str) -> re.Pattern:
    return re.compile(re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*"))


def _covered(path: str, patterns) -> bool:
    """GitHub path filters: the last pattern that matches decides, and `!` excludes."""
    covered = False
    for pattern in patterns:
        if _glob(pattern.lstrip("!")).fullmatch(path):
            covered = not pattern.startswith("!")
    return covered


def _rel(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return None


def _pester_reads() -> set[str]:
    found = set()
    for script in PESTER.rglob("*.ps1"):
        text = script.read_text(encoding = "utf-8")
        for rel in _FROM_SCRIPT_ROOT.findall(text):
            found.add(_rel(script.parent / rel.replace("\\", "/")))
        for rel in _FROM_SETUP_DIR.findall(text):
            found.add(_rel(STUDIO / rel.replace("\\", "/")))
    # $PSScriptRoot "venv" and similar scratch names resolve inside the suite's own directory.
    return {p for p in found if p and not (REPO / p).is_dir()}


def _resolver_imports() -> set[str]:
    """Repo modules install_llama_prebuilt.py imports, followed transitively."""
    roots = [STUDIO, REPO]
    seen, todo = set(), [STUDIO / "install_llama_prebuilt.py"]
    while todo:
        path = todo.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding = "utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            for name in names:
                parts = name.split(".")
                for root in roots:
                    for depth in range(len(parts), 0, -1):
                        base = root.joinpath(*parts[:depth])
                        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
                            if candidate.is_file():
                                todo.append(candidate)
    return {_rel(p) for p in seen if _rel(p)}


def test_the_scans_find_the_known_inputs():
    """A scan that matched nothing would pass the check below."""
    assert {"install.ps1", "studio/setup.ps1", "studio/install_node_prebuilt.py"} <= _pester_reads()
    assert {"studio/install_llama_prebuilt.py", "studio/prebuilt_core.py"} <= _resolver_imports()


def test_every_file_small_checks_reads_triggers_the_workflow():
    on = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))[True]
    inputs = _pester_reads() | _resolver_imports()
    for trigger in ("pull_request", "push"):
        patterns = on[trigger]["paths"]
        missing = sorted(p for p in inputs if not _covered(p, patterns))
        assert not missing, (
            f"small-checks reads these files, but no {trigger} path of {WORKFLOW.name} matches "
            f"them, so editing one would skip the Windows installer checks: {missing}"
        )
