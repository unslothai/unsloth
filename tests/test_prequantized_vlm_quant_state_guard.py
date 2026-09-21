# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""Tests for the transformers 5.4.0 / 5.5.x pre-quantized quant_state guard.

unsloth #9867, #10010, #10017, #10276. Those releases discard the bitsandbytes
quant_state sidecar tensors of pre-quantized composite (multimodal) checkpoints,
so every quantized Linear loads with ``quant_state = None`` and the first forward
raises ``mat1 and mat2 shapes cannot be multiplied``. Measured on one B200 with
``unsloth/qwen3.8-27b-unsloth-bnb-4bit``: 352 of 352 quantized Linear modules lose
quant_state on 5.4.0 and 5.5.4, and 0 of 352 on 5.2.0, 5.3.0 and every release from
5.6.2 to 5.17.0.

Loaded in isolation, no torch, no GPU, no network.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import pytest
import tomllib
from packaging.requirements import Requirement
from packaging.version import Version

_ROOT = Path(__file__).resolve().parent.parent
_IMPORT_FIXES_PATH = _ROOT / "unsloth" / "import_fixes.py"
_PYPROJECT_PATH = _ROOT / "pyproject.toml"

# The whole defect window, every release transformers actually published in it.
BROKEN = ["5.4.0", "5.5.0", "5.5.1", "5.5.2", "5.5.3", "5.5.4"]
# Neighbours on both sides, plus the 4.x floor and the current 5.x ceiling.
GOOD = ["4.57.6", "5.2.0", "5.3.0", "5.6.0", "5.6.2", "5.14.1", "5.17.0"]


def _load_import_fixes():
    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes_quant_state_under_test", _IMPORT_FIXES_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def import_fixes():
    return _load_import_fixes()


@pytest.mark.parametrize("version", BROKEN)
def test_broken_window_is_detected(import_fixes, version):
    assert import_fixes._transformers_drops_prequantized_vlm_quant_state(version) is True, (
        f"transformers {version} drops the bnb quant_state of pre-quantized composite "
        f"checkpoints and must be detected"
    )


@pytest.mark.parametrize("version", GOOD)
def test_good_versions_are_not_flagged(import_fixes, version):
    assert import_fixes._transformers_drops_prequantized_vlm_quant_state(version) is False, (
        f"transformers {version} loads pre-quantized composite checkpoints correctly and "
        f"must not be flagged"
    )


def test_window_boundaries_are_half_open(import_fixes):
    """5.4.0 inclusive (first broken), 5.6.0 exclusive (transformers PR #45567)."""
    low, high = import_fixes._BROKEN_PREQUANTIZED_VLM_TRANSFORMERS
    assert Version(low) == Version("5.4.0")
    assert Version(high) == Version("5.6.0")
    assert import_fixes._transformers_drops_prequantized_vlm_quant_state(low) is True
    assert import_fixes._transformers_drops_prequantized_vlm_quant_state(high) is False


def test_unparseable_version_never_raises(import_fixes):
    for junk in ("", "not-a-version", "5.x", None.__class__.__name__):
        assert import_fixes._transformers_drops_prequantized_vlm_quant_state(junk) is False


def test_warning_names_the_cause_and_the_remedy(import_fixes, monkeypatch, caplog):
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "5.5.4")
    monkeypatch.delenv("UNSLOTH_SKIP_TRANSFORMERS_QUANT_STATE_CHECK", raising = False)
    with caplog.at_level(logging.WARNING):
        import_fixes.check_transformers_prequantized_vlm_quant_state()
    text = caplog.text
    assert "5.5.4" in text
    assert "quant_state" in text
    # It must say the checkpoint is fine: the reported thread told users to regenerate it.
    assert "not be regenerated" in text.lower()
    assert "transformers>=5.6.0" in text


def test_warning_silent_on_a_good_version(import_fixes, monkeypatch, caplog):
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "5.17.0")
    monkeypatch.delenv("UNSLOTH_SKIP_TRANSFORMERS_QUANT_STATE_CHECK", raising = False)
    with caplog.at_level(logging.WARNING):
        import_fixes.check_transformers_prequantized_vlm_quant_state()
    assert "quant_state" not in caplog.text


def test_env_var_silences_the_warning(import_fixes, monkeypatch, caplog):
    monkeypatch.setattr(import_fixes, "importlib_version", lambda name: "5.5.4")
    monkeypatch.setenv("UNSLOTH_SKIP_TRANSFORMERS_QUANT_STATE_CHECK", "1")
    with caplog.at_level(logging.WARNING):
        import_fixes.check_transformers_prequantized_vlm_quant_state()
    assert "quant_state" not in caplog.text


def test_missing_transformers_never_raises(import_fixes, monkeypatch):
    def _boom(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(import_fixes, "importlib_version", _boom)
    import_fixes.check_transformers_prequantized_vlm_quant_state()


def _transformers_requirements():
    data = tomllib.loads(_PYPROJECT_PATH.read_text(encoding = "utf-8"))
    found = []
    for group in (data.get("project", {}).get("optional-dependencies", {}) or {}).values():
        for raw in group:
            try:
                req = Requirement(raw)
            except Exception:
                continue
            if req.name == "transformers":
                found.append(req)
    for raw in data.get("project", {}).get("dependencies", []) or []:
        try:
            req = Requirement(raw)
        except Exception:
            continue
        if req.name == "transformers":
            found.append(req)
    return found


def test_pyproject_excludes_every_broken_release():
    """The cap alone is not enough: 5.4.0 and 5.5.x must be excluded outright.

    Capping at <=5.5.0 leaves 5.4.0 and 5.5.0 installable and a fresh resolve takes
    5.5.0, dead centre of the window. These exclusions stay correct whenever the
    ceiling is raised later.
    """
    reqs = _transformers_requirements()
    assert reqs, "no transformers requirement found in pyproject.toml"
    for req in reqs:
        for version in BROKEN:
            assert not req.specifier.contains(version, prereleases = True), (
                f"pyproject allows transformers {version}, which discards the bnb "
                f"quant_state of pre-quantized composite checkpoints "
                f"(unsloth #9867, #10010, #10017, #10276)"
            )


def test_pyproject_still_allows_a_working_version():
    """Guard against fixing the above by excluding everything."""
    reqs = _transformers_requirements()
    assert reqs
    for req in reqs:
        assert any(
            req.specifier.contains(version, prereleases = True) for version in GOOD
        ), f"pyproject leaves no working transformers at all: {req}"
