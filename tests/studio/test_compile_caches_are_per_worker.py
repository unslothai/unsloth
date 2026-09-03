# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Parallel suites do not share one torch.compile cache directory.

Inductor's on-disk caches default to one directory per USER, not per process, so four
xdist workers on a runner would share `fxgraph`, `aotautograd` and the Triton cache
underneath it. The upstream recipe is explicit that a common `TORCHINDUCTOR_CACHE_DIR`
is how processes are made to SHARE compiled artifacts, so a different value per worker
is how they are kept apart.

Two things are asserted, and the second is the one that rots: that the splitting works,
and that it is actually reached from the conftest of every suite that runs with `-n`. A
helper nobody imports is the failure mode here, and it is silent.
"""

import importlib.util
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "tests" / "_shared" / "compile_cache_isolation.py"
CONFTESTS = (REPO / "tests" / "conftest.py", REPO / "studio" / "backend" / "tests" / "conftest.py")
WORKFLOW = REPO / ".github" / "workflows" / "studio-backend-ci.yml"


def _load():
    spec = importlib.util.spec_from_file_location("compile_cache_isolation_under_test", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_each_worker_gets_a_different_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    module = _load()

    seen = set()
    for worker in ("gw0", "gw1", "gw2", "gw3"):
        monkeypatch.setenv("PYTEST_XDIST_WORKER", worker)
        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
        module.isolate_compile_caches()
        seen.add(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    assert len(seen) == 4, f"four workers landed on {len(seen)} directories: {sorted(seen)}"


def test_triton_is_split_too(monkeypatch, tmp_path):
    """It only follows TORCHINDUCTOR_CACHE_DIR when unset, so an environment that
    exports it would otherwise keep every worker on one Triton cache."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TRITON_CACHE_DIR", "/somewhere/shared")
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw1")
    module = _load()
    module.isolate_compile_caches()
    assert os.environ["TRITON_CACHE_DIR"].startswith(os.environ["TORCHINDUCTOR_CACHE_DIR"]), (
        f"TRITON_CACHE_DIR is {os.environ['TRITON_CACHE_DIR']}, outside this worker's "
        f"inductor directory, so the Triton half is still shared"
    )


def test_a_single_process_run_is_left_alone(monkeypatch, tmp_path):
    """No xdist, no split: one process already has the default to itself, and moving it
    would drop whatever the environment deliberately pointed it at."""
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising = False)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    module = _load()
    assert module.isolate_compile_caches() is None
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(tmp_path)


def test_an_explicit_location_is_split_underneath_not_replaced(monkeypatch, tmp_path):
    """CI may point the cache at a path on purpose. The worker goes inside it."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "chosen"))
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    module = _load()
    module.isolate_compile_caches()
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"].startswith(
        str(tmp_path / "chosen")
    ), "an explicit TORCHINDUCTOR_CACHE_DIR was discarded rather than split underneath"


@pytest.mark.parametrize("conftest", CONFTESTS, ids = lambda p: str(p.relative_to(REPO)))
def test_every_parallel_suite_reaches_the_helper(conftest):
    """The quiet failure: the helper exists, nothing imports it, and the caches merge
    again with every test still green."""
    assert conftest.is_file(), f"{conftest} is gone"
    text = conftest.read_text(encoding = "utf-8")
    assert "compile_cache_isolation" in text, (
        f"{conftest.relative_to(REPO)} no longer loads the cache isolation helper, so its "
        f"workers share one inductor directory again. Nothing else would report it."
    )


def test_the_suites_this_protects_really_do_run_in_parallel():
    """If nothing runs with -n any more, this whole mechanism is dead weight and should
    be deleted rather than left looking load-bearing."""
    text = WORKFLOW.read_text(encoding = "utf-8")
    assert re.search(r"pytest[^\n]*-n\s+\d", text), (
        f"no parallel pytest invocation left in {WORKFLOW.name}; if the suites went back "
        f"to one process, remove the isolation helper instead of keeping it around"
    )
