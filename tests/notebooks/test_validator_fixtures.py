# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""
Golden-fixture tests for scripts/notebook_validator.py.

Each test reconstructs the broken-state install cell that one of the
referenced unslothai/notebooks PRs fixed, and asserts the matching rule
fires. The fixed-state tests prove the rule falls silent after the fix.

Cross-references:
  PR #258  -> R-INST-003  (peft/torchao floor)
  PR #260  -> R-EXC-001   (DONT_UPDATE_EXCEPTIONS coverage; covered by
                            an integration test pointing at a real
                            notebooks checkout)
  PR #261a -> R-INST-004  (torch/torchcodec ABI)
  PR #261b -> R-INST-005  (transformers --no-deps + tokenizers window)
  PR #264  -> R-INST-005  (same class as #261b)
  PR #221  -> R-INST-001  (forbid git+ HEAD installs)
  51b1462  -> R-DRIFT-001 (drift; integration-tested separately)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import notebook_validator as nv  # noqa: E402

# Snapshot of Colab GPU pip-freeze that recreates the bug environments
# below. Real CI uses scripts/data/colab_pip_freeze.gpu.txt; tests use a
# small inline subset so the unit cases are hermetic.
COLAB_2026_05 = {
    "torch": "2.10.0+cu128",
    "torchao": "0.10.0",
    "torchcodec": "0.10.0+cu128",
    "transformers": "5.0.0",
    "tokenizers": "0.22.2",
    "peft": "0.19.1",
    "accelerate": "1.13.0",
    "datasets": "4.0.0",
}


# ---------- R-INST-001 : forbid git+ HEAD ------------------------------- #


def test_r_inst_001_fires_on_transformers_git_head():
    cell = """%%capture
!pip install --force-reinstall git+https://github.com/huggingface/transformers.git
"""
    findings = nv.rule_inst_001_git_plus(cell, "fixture", 0)
    assert any(f.rule == "R-INST-001" for f in findings)


def test_r_inst_001_silent_after_pin():
    cell = """%%capture
!pip install transformers==5.5.0
"""
    findings = nv.rule_inst_001_git_plus(cell, "fixture", 0)
    assert findings == []


def test_r_inst_001_allowlist_unsloth_zoo_git():
    cell = """%%capture
!pip install --no-build-isolation git+https://github.com/state-spaces/mamba.git@main
!pip install "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo"
"""
    findings = nv.rule_inst_001_git_plus(cell, "fixture", 0)
    assert findings == []


# ---------- R-INST-003 : peft / torchao floor (PR #258) ------------------ #


def test_r_inst_003_fires_when_peft_19_with_no_torchao_bump():
    cell = """%%capture
!pip install --no-deps peft trl unsloth_zoo
"""
    findings = nv.rule_inst_003_peft_torchao(cell, COLAB_2026_05, "fixture", 0)
    assert any(f.rule == "R-INST-003" for f in findings)


def test_r_inst_003_silent_when_torchao_bumped():
    cell = """%%capture
!pip install --no-deps peft trl unsloth_zoo
!pip install --no-deps --upgrade "torchao>=0.16.0"
"""
    findings = nv.rule_inst_003_peft_torchao(cell, COLAB_2026_05, "fixture", 0)
    assert findings == []


def test_r_inst_003_silent_when_torchao_pinned_high():
    cell = """%%capture
!pip install --no-deps peft trl
!pip install torchao==0.17.0
"""
    findings = nv.rule_inst_003_peft_torchao(cell, COLAB_2026_05, "fixture", 0)
    assert findings == []


# ---------- R-INST-004 : torch / torchcodec ABI (PR #261a) --------------- #


def test_r_inst_004_fires_torch_2_7_with_torchcodec_0_6():
    cell = """%%capture
!uv pip install "torch==2.7.1"
!uv pip install --no-deps "torchcodec==0.6.0"
"""
    findings = nv.rule_inst_004_torchcodec_torch(cell, COLAB_2026_05, "fixture", 0)
    assert any(f.rule == "R-INST-004" for f in findings)


def test_r_inst_004_silent_when_torch_2_7_with_torchcodec_0_5():
    cell = """%%capture
!uv pip install "torch==2.7.1"
!uv pip install --no-deps "torchcodec==0.5"
"""
    findings = nv.rule_inst_004_torchcodec_torch(cell, COLAB_2026_05, "fixture", 0)
    assert findings == []


# ---------- R-INST-005 : transformers + tokenizers window (PRs #261b/#264) -- #


def test_r_inst_005_fires_no_deps_transformers_55_without_tokenizers_pin(monkeypatch):
    """PR #264: --no-deps transformers==5.5.0 leaves Colab tokenizers in
    place; if Colab ever ships tokenizers > 0.23.0 this breaks."""
    cell = """%%capture
!pip install --no-deps transformers==5.5.0
"""
    # Fake a Colab snapshot where tokenizers has just bumped past the window
    # transformers 5.5.0 supports.
    colab = dict(COLAB_2026_05, tokenizers = "0.23.5")

    def fake_meta(name, version):
        if name.lower() == "transformers" and version == "5.5.0":
            return {"info": {"requires_dist": ["tokenizers (>=0.22.0,<=0.23.0)"]}}
        return None

    monkeypatch.setattr(nv, "pypi_metadata", fake_meta)

    findings = nv.rule_inst_005_transformers_tokenizers(cell, colab, "fixture", 0)
    assert any(f.rule == "R-INST-005" for f in findings)


def test_r_inst_005_silent_when_no_deps_pins_tokenizers(monkeypatch):
    cell = """%%capture
!pip install --no-deps transformers==5.5.0 "tokenizers>=0.22.0,<=0.23.0"
"""

    def fake_meta(name, version):
        if name.lower() == "transformers" and version == "5.5.0":
            return {"info": {"requires_dist": ["tokenizers (>=0.22.0,<=0.23.0)"]}}
        return None

    monkeypatch.setattr(nv, "pypi_metadata", fake_meta)
    # Cell wins over Colab; resolved tokenizers will be 0.23.0.
    colab = dict(COLAB_2026_05, tokenizers = "0.23.5")

    findings = nv.rule_inst_005_transformers_tokenizers(cell, colab, "fixture", 0)
    assert findings == []


def test_r_inst_005_silent_without_no_deps(monkeypatch):
    """If --no-deps is absent, pip resolves tokenizers transitively; the
    rule must NOT fire (this is the false-positive case from notebooks like
    Whisper.ipynb that pin transformers but rely on pip's resolver)."""
    cell = """%%capture
!pip install transformers==4.51.3
"""

    def fake_meta(name, version):
        if name.lower() == "transformers" and version == "4.51.3":
            return {"info": {"requires_dist": ["tokenizers (>=0.21,<0.22)"]}}
        return None

    monkeypatch.setattr(nv, "pypi_metadata", fake_meta)
    colab = COLAB_2026_05
    findings = nv.rule_inst_005_transformers_tokenizers(cell, colab, "fixture", 0)
    assert findings == []


# ---------- R-API-003 : suboptimal optim warning (PR #221, partial) ------ #

import json
from pathlib import Path as _P


def _nb_with_code(*sources: str) -> dict:
    return {
        "cells": [{"cell_type": "code", "source": s} for s in sources],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_r_api_003_fires_on_adamw_torch_fused():
    nb = _nb_with_code(
        "%%capture\n!pip install unsloth\n",
        'from trl import SFTConfig\ntrainer = SFTConfig(optim="adamw_torch_fused")\n',
    )
    findings = nv.scan_user_cells(nb, "fixture")
    assert any(f.rule == "R-API-003" for f in findings)


def test_r_api_003_silent_on_adamw_8bit():
    nb = _nb_with_code(
        "%%capture\n!pip install unsloth\n",
        'from trl import SFTConfig\ntrainer = SFTConfig(optim="adamw_8bit")\n',
    )
    findings = nv.scan_user_cells(nb, "fixture")
    assert findings == []


# ---------- Environment classifier --------------------------------------- #


@pytest.mark.parametrize(
    "path,expected",
    [
        ("nb/Llama3.1_(8B)-Alpaca.ipynb", "colab"),
        ("nb/Kaggle-Llama3.1_(8B)-Alpaca.ipynb", "kaggle"),
        ("kaggle/Gemma4_(31B)-Text.ipynb", "kaggle"),
        ("nb/AMD-Llama3.1_(8B)-Alpaca.ipynb", "amd"),
        ("nb/HuggingFace Course-Qwen3_(4B)-GRPO.ipynb", "colab"),
        (
            "nb/gpt_oss_(20B)_Reinforcement_Learning_2048_Game_DGX_Spark.ipynb",
            "dgx_spark",
        ),
    ],
)
def test_environment_classifier(path, expected):
    assert nv.target_environment(path) == expected


# ---------- Integration: walk the live notebooks repo (skipped if absent) -- #


def _live_notebooks_dir() -> Path | None:
    candidates = [
        Path(__file__).resolve().parents[3] / "notebooks",  # workspace sibling
        Path("/mnt/disks/unslothai/ubuntu/workspace_12/notebooks"),
    ]
    for p in candidates:
        if (p / "update_all_notebooks.py").is_file():
            return p
    return None


@pytest.mark.skipif(
    _live_notebooks_dir() is None,
    reason = "unslothai/notebooks not cloned at sibling path",
)
def test_exceptions_passes_on_head():
    """L1.2 must be silent on the live HEAD of unslothai/notebooks. If this
    test fires, either DONT_UPDATE_EXCEPTIONS gained a notebook missing a
    policy clause (real bug) or the policy clause set is stale."""
    findings = nv.rule_l12_exceptions_coverage(_live_notebooks_dir())
    assert findings == [], findings


@pytest.mark.skipif(
    _live_notebooks_dir() is None,
    reason = "unslothai/notebooks not cloned at sibling path",
)
def test_lint_smoke_no_module_errors():
    """The lint subcommand should walk every nb/kaggle without crashing.
    (We accept findings -- those are the validator doing its job.)"""
    import subprocess

    rc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "notebook_validator.py"),
            "lint",
            "--no-pypi",
            "--notebooks-dir",
            str(_live_notebooks_dir()),
            "--colab-pin",
            str(SCRIPTS_DIR / "data" / "colab_pip_freeze.gpu.txt"),
        ],
        capture_output = True,
        text = True,
        timeout = 120,
    )
    # rc=0 means clean, rc=1 means findings reported, rc=2 means crash.
    assert rc.returncode in (0, 1), rc.stderr[-2000:]
