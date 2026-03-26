"""End-to-end tests: Studio modules import without torch in an isolated venv."""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_COLLATORS = (
    REPO_ROOT / "studio" / "backend" / "utils" / "datasets" / "data_collators.py"
)
CHAT_TEMPLATES = (
    REPO_ROOT / "studio" / "backend" / "utils" / "datasets" / "chat_templates.py"
)


def _has_uv() -> bool:
    return shutil.which("uv") is not None


@pytest.fixture(scope = "module")
def no_torch_venv(tmp_path_factory):
    """Create a temporary venv with no torch installed. Returns the python path."""
    if not _has_uv():
        pytest.skip("uv not available")

    venv_dir = tmp_path_factory.mktemp("no_torch_venv")
    # Try 3.12 first (always available), fall back to system python
    result = subprocess.run(
        ["uv", "venv", str(venv_dir), "--python", "3.12"],
        capture_output = True,
    )
    if result.returncode != 0:
        result = subprocess.run(
            ["uv", "venv", str(venv_dir)],
            capture_output = True,
        )
        if result.returncode != 0:
            pytest.skip(f"Could not create venv: {result.stderr.decode()}")

    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():
        venv_python = venv_dir / "Scripts" / "python.exe"  # Windows
    if not venv_python.exists():
        pytest.skip("Could not find python in venv")

    # Verify torch is NOT importable
    check = subprocess.run(
        [str(venv_python), "-c", "import torch"],
        capture_output = True,
    )
    assert check.returncode != 0, "torch should NOT be importable in the test venv"

    return str(venv_python)


# ── data_collators.py tests ─────────────────────────────────────────────


class TestDataCollatorsNoTorch:
    """Verify data_collators.py can be parsed and loaded without torch."""

    def test_ast_parse(self):
        """data_collators.py must be valid Python syntax."""
        source = DATA_COLLATORS.read_text(encoding = "utf-8")
        tree = ast.parse(source, filename = str(DATA_COLLATORS))
        assert tree is not None

    def test_no_top_level_torch_import(self):
        """No top-level 'import torch' or 'from torch' statements."""
        source = DATA_COLLATORS.read_text(encoding = "utf-8")
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(
                        "torch"
                    ), f"Top-level 'import {alias.name}' found at line {node.lineno}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith(
                        "torch"
                    ), f"Top-level 'from {node.module}' found at line {node.lineno}"

    def test_exec_in_no_torch_venv(self, no_torch_venv):
        """data_collators.py executes in a venv without torch (with loggers stub)."""
        result = subprocess.run(
            [
                no_torch_venv,
                "-c",
                "import sys, types; "
                "loggers = types.ModuleType('loggers'); "
                "loggers.get_logger = lambda n: None; "
                "sys.modules['loggers'] = loggers; "
                f"exec(open({str(DATA_COLLATORS)!r}).read())",
            ],
            capture_output = True,
            timeout = 30,
        )
        assert (
            result.returncode == 0
        ), f"data_collators.py failed in no-torch venv:\n{result.stderr.decode()}"


# ── chat_templates.py tests ─────────────────────────────────────────────


class TestChatTemplatesNoTorch:
    """Verify chat_templates.py has no top-level torch imports."""

    def test_ast_parse(self):
        """chat_templates.py must be valid Python syntax."""
        source = CHAT_TEMPLATES.read_text(encoding = "utf-8")
        tree = ast.parse(source, filename = str(CHAT_TEMPLATES))
        assert tree is not None

    def test_no_top_level_torch_import(self):
        """No top-level 'import torch' or 'from torch' at module level."""
        source = CHAT_TEMPLATES.read_text(encoding = "utf-8")
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(
                        "torch"
                    ), f"Top-level 'import {alias.name}' found at line {node.lineno}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not node.module.startswith(
                        "torch"
                    ), f"Top-level 'from {node.module}' found at line {node.lineno}"

    def test_torch_imports_only_inside_functions(self):
        """All 'from torch' imports must be inside function/method bodies."""
        source = CHAT_TEMPLATES.read_text(encoding = "utf-8")
        tree = ast.parse(source)
        torch_imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = None
                if isinstance(node, ast.ImportFrom):
                    module = node.module
                elif isinstance(node, ast.Import):
                    module = node.names[0].name if node.names else None
                if module and module.startswith("torch"):
                    torch_imports.append(node)

        # Each torch import should NOT be a direct child of the module
        top_level = set(id(n) for n in ast.iter_child_nodes(tree))
        for imp in torch_imports:
            assert (
                id(imp) not in top_level
            ), f"torch import at line {imp.lineno} is at top level (should be inside a function)"


# ── Negative control ────────────────────────────────────────────────────


class TestNegativeControl:
    """Confirm that prepending 'import torch' DOES fail in the no-torch venv."""

    def test_import_torch_fails(self, no_torch_venv):
        """Adding 'import torch' at the top should cause ImportError."""
        # Create a temp file with import torch prepended
        source = DATA_COLLATORS.read_text(encoding = "utf-8")
        with tempfile.NamedTemporaryFile(
            mode = "w", suffix = ".py", delete = False, encoding = "utf-8"
        ) as f:
            f.write("import torch\n")
            f.write(source)
            temp_file = f.name

        try:
            result = subprocess.run(
                [
                    no_torch_venv,
                    "-c",
                    "import sys, types; "
                    "loggers = types.ModuleType('loggers'); "
                    "loggers.get_logger = lambda n: None; "
                    "sys.modules['loggers'] = loggers; "
                    f"exec(open({temp_file!r}).read())",
                ],
                capture_output = True,
                timeout = 30,
            )
            assert (
                result.returncode != 0
            ), "Expected failure when 'import torch' is prepended"
            assert (
                b"ModuleNotFoundError" in result.stderr
                or b"ImportError" in result.stderr
            ), f"Expected ImportError, got:\n{result.stderr.decode()}"
        finally:
            os.unlink(temp_file)
