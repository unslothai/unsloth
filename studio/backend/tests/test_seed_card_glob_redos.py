# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Hub dataset card's data_files glob must not stall the shared backend: `**a**a**a...`
backtracked exponentially as a regex, holding the GIL while one preview ran."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent


def _run(code: str, timeout: float) -> str:
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd = BACKEND,
        capture_output = True,
        text = True,
        timeout = timeout,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip()


def test_card_glob_matching_is_not_exponential():
    pytest.importorskip("fastapi")
    pytest.importorskip("structlog")
    code = """
        from routes.data_recipe.seed import _files_under_patterns
        files = ["a" * 200 + ".json", "data/train-0.json"]
        print(_files_under_patterns(["**a" * 40 + "b"], files, as_the_loader_would = True))
    """
    try:
        assert _run(code, timeout = 60) == "[]"
    except subprocess.TimeoutExpired:
        pytest.fail("card glob matching backtracked past 60 s")


def test_card_glob_matches_are_unchanged():
    pytest.importorskip("fastapi")
    pytest.importorskip("structlog")
    code = """
        from routes.data_recipe.seed import _glob_to_regex
        cases = [
            ("data/**/train-*.parquet", "data/x/y/train-0.parquet"),
            ("data/**/train-*.parquet", "data/train-0.parquet"),
            ("data/**/train.*", "data/nottrain.x"),
            ("**/*.json", "a/b.json"),
            ("*.json", "a/b.json"),
            ("train-??.csv", "train-01.csv"),
            ("train-[!0].csv", "train-1.csv"),
            ("train-[!0].csv", "train-0.csv"),
            ("[^x]a.csv", "^a.csv"),
            ("a**", "a/b/c"),
            ("data/*", "data/x/y"),
        ]
        print([bool(_glob_to_regex(p).match(s)) for p, s in cases])
    """
    assert _run(code, timeout = 60) == str(
        [True, True, False, True, False, True, True, False, True, True, False]
    )
