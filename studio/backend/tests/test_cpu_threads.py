# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for Studio's early CPU thread-pool configuration."""

from pathlib import Path

import pytest

from utils.cpu_threads import configure_cpu_threads


_NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_RUN_PY = Path(__file__).resolve().parent.parent / "run.py"


def test_cpu_thread_cap_seeds_native_pool_limits():
    env = {"UNSLOTH_CPU_THREADS": " 6 "}

    configure_cpu_threads(env)

    assert {variable: env[variable] for variable in _NATIVE_THREAD_VARIABLES} == {
        variable: "6" for variable in _NATIVE_THREAD_VARIABLES
    }


def test_cpu_thread_cap_preserves_runtime_specific_override():
    env = {"UNSLOTH_CPU_THREADS": "4", "OMP_NUM_THREADS": "2"}

    configure_cpu_threads(env)

    assert env["OMP_NUM_THREADS"] == "2"
    assert env["MKL_NUM_THREADS"] == "4"


def test_cpu_thread_cap_is_opt_in():
    env = {}

    configure_cpu_threads(env)

    assert all(variable not in env for variable in _NATIVE_THREAD_VARIABLES)


@pytest.mark.parametrize("configured", ["zero", "0", "-3"])
def test_cpu_thread_cap_requires_positive_integer(configured):
    with pytest.raises(ValueError, match = "must be a positive integer"):
        configure_cpu_threads({"UNSLOTH_CPU_THREADS": configured})


def test_cpu_thread_configuration_runs_before_backend_imports():
    source = _RUN_PY.read_text()

    assert source.index("configure_cpu_threads()") < source.index(
        "import _platform_compat"
    )
