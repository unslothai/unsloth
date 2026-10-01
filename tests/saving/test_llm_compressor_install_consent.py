# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FP8/FP4 export installs a missing llm-compressor only with install_missing_dependencies=True (#8904)."""

import importlib.abc
import inspect
import sys

import pytest

import unsloth.save as save


class _NoLlmCompressor(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        name,
        path,
        target = None,
    ):
        if name == "llmcompressor" or name.startswith("llmcompressor."):
            raise ImportError("simulated missing llm-compressor")


@pytest.fixture
def pip_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL", raising = False)
    monkeypatch.setattr(save, "_llm_compressor_imports_in_subprocess", lambda: False)
    calls = []
    monkeypatch.setattr(save.subprocess, "check_call", lambda cmd, *a, **k: calls.append(cmd))
    blocker = _NoLlmCompressor()
    sys.meta_path.insert(0, blocker)
    yield calls
    sys.meta_path.remove(blocker)


def test_default_raises_with_manual_command_and_never_installs(pip_calls):
    with pytest.raises(RuntimeError) as e:
        save.install_llm_compressor()
    assert pip_calls == []
    assert save.llm_compressor_manual_install_command() in str(e.value)
    assert "install_missing_dependencies=True" in str(e.value)


def test_consent_runs_the_pinned_install(pip_calls):
    with pytest.raises(RuntimeError, match = "installed but could not be imported"):
        save.install_llm_compressor(install_missing_dependencies = True)
    assert len(pip_calls) == 1
    assert save._LLM_COMPRESSOR_SPEC in pip_calls[0]


def test_env_optout_beats_consent(pip_calls, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL", "1")
    with pytest.raises(RuntimeError, match = "UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL") as e:
        save.install_llm_compressor(install_missing_dependencies = True)
    assert pip_calls == []
    assert "install_missing_dependencies=True" not in str(e.value)


@pytest.mark.parametrize(
    "fn",
    [
        save.unsloth_save_pretrained_merged,
        save.unsloth_push_to_hub_merged,
        save.unsloth_generic_save_pretrained_merged,
        save.unsloth_generic_push_to_hub_merged,
        save._unsloth_save_compressed_tensors,
    ],
)
def test_entrypoints_default_to_no_install(fn):
    assert inspect.signature(fn).parameters["install_missing_dependencies"].default is False
