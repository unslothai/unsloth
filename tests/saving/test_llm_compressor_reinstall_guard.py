# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Must not pip-reinstall llm-compressor when only this process fails to import it."""

import importlib.abc
import subprocess
import sys

import pytest

import unsloth.save as save


def _write_fake(
    root,
    version = "0.11.0",
    broken = False,
):
    pkg = root / "llmcompressor"
    (pkg / "modifiers" / "quantization").mkdir(parents = True, exist_ok = True)
    body = "raise ImportError('broken install')\n" if broken else ""
    (pkg / "__init__.py").write_text(f"{body}__version__ = {version!r}\ndef oneshot(**kw): pass\n")
    (pkg / "modifiers" / "__init__.py").write_text("")
    (pkg / "modifiers" / "quantization" / "__init__.py").write_text(
        "class QuantizationModifier: pass\n"
    )


class _BlockInProcess(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        name,
        path,
        target = None,
    ):
        if name == "llmcompressor" or name.startswith("llmcompressor."):
            raise ImportError("simulated in-process failure")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL", raising = False)
    blocker = _BlockInProcess()
    sys.meta_path.insert(0, blocker)
    pip_calls = []
    monkeypatch.setattr(save.subprocess, "check_call", lambda cmd, *a, **k: pip_calls.append(cmd))
    yield tmp_path, pip_calls
    sys.meta_path.remove(blocker)


@pytest.mark.parametrize("optout", ["0", "1"])
def test_installed_but_blocked_in_process_skips_pip(env, monkeypatch, optout):
    root, pip_calls = env
    monkeypatch.setenv("UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL", optout)
    _write_fake(root)
    assert save.install_llm_compressor() == (None, None)
    assert pip_calls == []


@pytest.mark.parametrize("fake", [dict(broken = True), dict(version = "0.99.0")])
def test_broken_or_out_of_range_install_is_still_reinstalled(env, fake):
    root, pip_calls = env
    _write_fake(root, **fake)
    with pytest.raises(RuntimeError, match = "installed but could not be imported"):
        save.install_llm_compressor()
    assert len(pip_calls) == 1


def test_install_that_the_subprocess_can_use_is_accepted(env, monkeypatch):
    root, pip_calls = env
    _write_fake(root, broken = True)
    monkeypatch.setattr(
        save.subprocess,
        "check_call",
        lambda cmd, *a, **k: (pip_calls.append(cmd), _write_fake(root)),
    )
    assert save.install_llm_compressor() == (None, None)
    assert len(pip_calls) == 1


def test_probe_timeout_is_not_a_yes(monkeypatch):
    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired("python", 600)

    monkeypatch.setattr(save.subprocess, "run", _timeout)
    assert save._llm_compressor_imports_in_subprocess() is False


def test_relative_pythonpath_resolves_against_the_callers_cwd(env, monkeypatch):
    root, pip_calls = env
    _write_fake(root / "vendor")
    monkeypatch.chdir(root)
    monkeypatch.setenv("PYTHONPATH", "vendor")
    assert save.install_llm_compressor() == (None, None)
    assert pip_calls == []
