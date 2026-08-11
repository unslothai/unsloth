# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The AOTriton gate: opened by default, never argued with when already set."""

from utils.rocm_attention import AOTRITON_ENV, enable_rocm_aotriton_attention


def test_opens_the_gate_when_unset():
    env = {}
    assert enable_rocm_aotriton_attention(env) is True
    assert env[AOTRITON_ENV] == "1"


def test_zero_is_the_opt_out_and_survives():
    """ "0" is a deliberate choice by someone who hit an AOTriton bug, not an empty slot."""
    env = {AOTRITON_ENV: "0"}
    assert enable_rocm_aotriton_attention(env) is False
    assert env[AOTRITON_ENV] == "0"


def test_existing_value_is_never_rewritten():
    env = {AOTRITON_ENV: "whatever-the-operator-set"}
    assert enable_rocm_aotriton_attention(env) is False
    assert env[AOTRITON_ENV] == "whatever-the-operator-set"


def test_idempotent():
    """run.py may be imported as well as executed; the second call must not re-decide."""
    env = {}
    assert enable_rocm_aotriton_attention(env) is True
    assert enable_rocm_aotriton_attention(env) is False
    assert env[AOTRITON_ENV] == "1"


def test_defaults_to_the_real_environment(monkeypatch):
    monkeypatch.delenv(AOTRITON_ENV, raising = False)
    import os

    assert enable_rocm_aotriton_attention() is True
    assert os.environ[AOTRITON_ENV] == "1"


def test_run_py_opens_the_gate_before_importing_torch():
    """The whole fix is ordering: torch latches the var at import, so a late set is dead code."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "run.py").read_text(encoding = "utf-8")
    gate = src.index("enable_rocm_aotriton_attention()")
    # Nothing before the gate may pull torch, directly or via the stub installers.
    assert "import torch" not in src[:gate]
    assert src.index("from utils.cpu_threads") > gate
    assert src.index("install_torchao_windows_rocm_stub()") > gate
