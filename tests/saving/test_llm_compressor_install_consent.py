# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FP8/FP4 export auto-installs a missing llm-compressor unless install_missing_dependencies=False or the env opt-out is set (#8904)."""

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


def test_default_runs_the_pinned_install(pip_calls):
    with pytest.raises(RuntimeError, match = "installed but could not be imported"):
        save.install_llm_compressor()
    assert len(pip_calls) == 1
    assert save._LLM_COMPRESSOR_SPEC in pip_calls[0]


def test_opt_out_raises_with_manual_command_and_never_installs(pip_calls):
    with pytest.raises(RuntimeError) as e:
        save.install_llm_compressor(install_missing_dependencies = False)
    assert pip_calls == []
    assert save.llm_compressor_manual_install_command() in str(e.value)
    assert "install_missing_dependencies=False" in str(e.value)


@pytest.mark.parametrize("has_pip", [True, False])
def test_manual_command_matches_the_available_installer(monkeypatch, has_pip):
    import importlib.util

    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: (find_spec(name, *a, **k) if has_pip else None)
        if name == "pip"
        else find_spec(name, *a, **k),
    )
    command = save.llm_compressor_manual_install_command()
    assert command.startswith(f"{sys.executable} -m pip install" if has_pip else "uv pip install")


def test_env_optout_blocks_install(pip_calls, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL", "1")
    with pytest.raises(RuntimeError, match = "UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL") as e:
        save.install_llm_compressor(install_missing_dependencies = True)
    assert pip_calls == []
    assert "install_missing_dependencies" not in str(e.value)


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
def test_entrypoints_default_to_install(fn):
    assert inspect.signature(fn).parameters["install_missing_dependencies"].default is True
