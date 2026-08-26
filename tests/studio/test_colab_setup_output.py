# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the Colab setup cell's installer diagnostics visible on failure."""

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "studio" / "Unsloth_Studio_Colab.ipynb"


def _setup_cell() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding = "utf-8"))
    cells = (
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    return next(cell for cell in cells if "./studio/setup.sh" in cell)


def test_colab_setup_prints_combined_installer_output_before_failing():
    source = _setup_cell()

    assert "%cd /content/unsloth" in source
    assert '["./studio/setup.sh", "--local"]' in source
    assert "stdout=subprocess.PIPE" in source
    assert "stderr=subprocess.STDOUT" in source
    assert 'print(setup.stdout, end="")' in source
    assert "if setup.returncode:" in source
    error = 'raise RuntimeError(f"Studio setup failed with exit code {setup.returncode}")'
    assert error in source
    assert source.index('print(setup.stdout, end="")') < source.index(error)
    assert "check=True" not in source
