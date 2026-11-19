# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import io
import re
from contextlib import redirect_stdout

import pytest

from core.inference import tools


@pytest.fixture
def output_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tools, "_spill_records_dir", lambda: str(tmp_path / "records"))
    monkeypatch.setattr(tools, "_posix_tools_available", lambda: False)
    token = tools._REQUEST_RESULT_BUDGET.set(None)
    yield tmp_path
    tools._REQUEST_RESULT_BUDGET.reset(token)


@pytest.mark.parametrize(
    "text",
    [
        "".join(f"row {i:03d}: café 🦥\n" for i in range(100)),
        "漢字🦥é" * 300,
    ],
)
def test_python_paging_reads_the_next_unseen_chunk_without_a_shell(output_dir, text):
    result = tools._truncate(text, limit = 300, workdir = str(output_dir), reader = "python")
    assert "continue with:" in result
    prefix = result.split("\n\n... (truncated", 1)[0]
    code = result.split(" -- continue with:\n  ", 1)[1][:-1]
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exec(code, {})
    following = stdout.getvalue()
    assert following
    assert text.startswith(prefix + following) or text.startswith(prefix + "\n" + following)
    assert "sed -n" not in code


def test_complete_output_is_unchanged(output_dir):
    assert (
        tools._truncate("complete\n", limit = 300, workdir = str(output_dir), reader = "python")
        == "complete\n"
    )
