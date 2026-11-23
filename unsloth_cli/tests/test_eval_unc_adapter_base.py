# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth eval` on a Hub adapter must not hand its base_model_name_or_path to a Windows network share."""

from __future__ import annotations

import json

import pytest

import unsloth_cli.commands.eval as evalmod
from unsloth_cli.tests.test_eval import _run, env  # noqa: F401  (fixture)

UNC_BASES = [
    "\\\\attacker.example\\share\\model",
    "//attacker.example/share/model",
    "\\\\?\\UNC\\attacker.example\\share\\model",
    "file://attacker.example/share/model",
]


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(evalmod, "_WINDOWS", True, raising = False)


@pytest.mark.parametrize("backend", ["hf", "unsloth"])
@pytest.mark.parametrize("base", UNC_BASES)
def test_hub_adapter_with_unc_base_is_refused(env, windows, monkeypatch, tmp_path, base, backend):
    monkeypatch.setattr(evalmod, "resolve_base_model", lambda model: base)
    monkeypatch.setattr(evalmod, "_base_vocab_size", lambda m: pytest.fail("base config read"))
    result = _run(
        "attacker/adapter", "--tasks", "gsm8k", "--backend", backend, "-o", str(tmp_path / "o")
    )
    assert result.exit_code != 0, result.output
    assert "network share" in result.output
    assert "model" not in env and "model_name" not in env


@pytest.mark.parametrize(
    "base", ["org/base", "C:\\models\\base", "\\\\wsl$\\Ubuntu\\m", "\\\\?\\C:\\models\\base"]
)
def test_hub_adapter_with_ordinary_base_still_loads(env, windows, monkeypatch, tmp_path, base):
    monkeypatch.setattr(evalmod, "resolve_base_model", lambda model: base)
    result = _run("org/adapter", "--tasks", "gsm8k", "--backend", "hf", "-o", str(tmp_path / "o"))
    assert result.exit_code == 0, result.output
    assert env["model_args"]["pretrained"] == base


def test_local_adapter_and_explicit_base_are_the_users_own(env, windows, tmp_path):
    adapter = tmp_path / "lora"
    adapter.mkdir()
    unc = "\\\\fileserver\\models\\base"
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": unc}))
    result = _run(str(adapter), "--tasks", "gsm8k", "--backend", "hf", "-o", str(tmp_path / "a"))
    assert result.exit_code == 0, result.output
    assert env["model_args"]["pretrained"] == unc
    result = _run(
        "org/adapter",
        "--tasks",
        "gsm8k",
        "--backend",
        "hf",
        "--base-model",
        unc,
        "-o",
        str(tmp_path / "b"),
    )
    assert result.exit_code == 0, result.output


def test_off_windows_nothing_changes(env, monkeypatch, tmp_path):
    monkeypatch.setattr(evalmod, "_WINDOWS", False, raising = False)
    monkeypatch.setattr(evalmod, "resolve_base_model", lambda model: "//host/share/m")
    result = _run("org/adapter", "--tasks", "gsm8k", "--backend", "hf", "-o", str(tmp_path / "o"))
    assert result.exit_code == 0, result.output
