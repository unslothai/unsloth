# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unsloth#3551: `fast_inference = True` (vLLM) is refused under FSDP2, before vLLM loads."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from unsloth.models import loader_utils
from unsloth.models.loader_utils import fsdp2_requested, raise_if_fast_inference_under_fsdp2


@pytest.fixture
def no_launcher(monkeypatch):
    monkeypatch.delenv("FSDP_VERSION", raising = False)
    state = pytest.importorskip("accelerate.state")
    monkeypatch.setattr(state.AcceleratorState, "_shared_state", {})
    return monkeypatch


@pytest.mark.parametrize("value, fsdp2", [("2", True), ("1", False), ("0", False)])
def test_the_launcher_env_names_fsdp2(no_launcher, value, fsdp2):
    no_launcher.setenv("FSDP_VERSION", value)
    assert fsdp2_requested() is fsdp2


def test_an_accelerator_with_an_fsdp2_plugin(no_launcher):
    from accelerate.state import AcceleratorState
    no_launcher.setattr(
        AcceleratorState, "_shared_state", {"fsdp_plugin": SimpleNamespace(fsdp_version = 2)}
    )
    assert fsdp2_requested()


def test_vllm_under_fsdp2_is_refused_with_the_way_out(no_launcher):
    no_launcher.setenv("FSDP_VERSION", "2")
    with pytest.raises(NotImplementedError, match = "fast_inference = False"):
        raise_if_fast_inference_under_fsdp2(True)


@pytest.mark.parametrize("fsdp_version, fast_inference", [("2", False), ("1", True), (None, True)])
def test_everything_else_loads(no_launcher, fsdp_version, fast_inference):
    if fsdp_version is not None:
        no_launcher.setenv("FSDP_VERSION", fsdp_version)
    raise_if_fast_inference_under_fsdp2(fast_inference)


def test_both_loaders_refuse_before_looking_for_vllm():
    """FastLanguageModel and FastModel: the guard runs before the vLLM availability check."""
    source = Path(loader_utils.__file__).with_name("loader.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)
    checked = []
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for fn in (
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained"
        ):
            calls = [
                (n.lineno, ast.unparse(n.func))
                for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and ast.unparse(n.func)
                in ("raise_if_fast_inference_under_fsdp2", "_vllm_unavailable_error")
            ]
            names = [name for _, name in sorted(calls)]
            if "_vllm_unavailable_error" in names:
                assert names[0] == "raise_if_fast_inference_under_fsdp2", (cls.name, names)
                checked.append(cls.name)
    assert {"FastLanguageModel", "FastModel"} <= set(checked), checked
