# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PAL_DISABLE_SDMA default for Windows ROCm processes (rocm-systems#12571). CPU only, no torch import."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from utils import desktop_shell_env as dse
from utils.rocm_pal_env import PAL_DISABLE_SDMA_ENV, apply_windows_rocm_pal_defaults

ROCM = lambda: "2.11.0+rocm7.14.0"  # noqa: E731
CUDA = lambda: "2.11.0+cu128"  # noqa: E731


@pytest.mark.parametrize(
    "platform, version, expected",
    [
        ("win32", ROCM, "1"),
        ("win32", lambda: "2.11.0+ROCm7.13.0", "1"),
        ("win32", CUDA, None),
        ("win32", lambda: "2.11.0", None),
        ("win32", lambda: None, None),
        ("linux", ROCM, None),
        ("darwin", ROCM, None),
    ],
)
def test_default_only_for_windows_rocm(platform, version, expected):
    env: dict = {}
    name = apply_windows_rocm_pal_defaults(env, platform = platform, torch_version = version)
    assert env.get(PAL_DISABLE_SDMA_ENV) == expected
    assert name == (PAL_DISABLE_SDMA_ENV if expected else None)


@pytest.mark.parametrize("value", ["0", "1", ""])
def test_explicit_value_wins(value):
    env = {PAL_DISABLE_SDMA_ENV: value}
    assert apply_windows_rocm_pal_defaults(env, platform = "win32", torch_version = ROCM) is None
    assert env == {PAL_DISABLE_SDMA_ENV: value}


def test_never_raises():
    def boom():
        raise RuntimeError("metadata broken")

    env: dict = {}
    assert apply_windows_rocm_pal_defaults(env, platform = "win32", torch_version = boom) is None
    assert env == {}


def test_shell_override_is_preserved():
    assert PAL_DISABLE_SDMA_ENV in dse.ROCM_SHELL_ENV_ALLOWLIST


def test_main_applies_it_before_torch():
    """main.py calls the helper at module scope, before its first torch import."""
    src = (Path(__file__).resolve().parent.parent / "main.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    call_line = torch_line = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "_apply_windows_rocm_pal_defaults"
        ):
            call_line = node.lineno if call_line is None else min(call_line, node.lineno)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            )
            if any(n == "torch" or n.startswith("torch.") for n in names):
                torch_line = node.lineno if torch_line is None else min(torch_line, node.lineno)
    assert call_line is not None
    assert torch_line is None or call_line < torch_line
