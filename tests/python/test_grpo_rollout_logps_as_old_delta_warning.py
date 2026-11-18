# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""The generated GRPOConfig warns when UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD=1 is set without the dual clip
`delta`: the PPO ratio is then pi_theta / pi_rollout, unbounded on negative-advantage tokens."""

from __future__ import annotations

import ast
import contextlib
import io
import os
import re

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")


def _load(name):
    with open(SOURCE_PATH, "r", encoding = "utf-8") as fh:
        tree = ast.parse(fh.read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"re": re, "os": os}
    exec(compile(ast.Module(body = [node], type_ignores = []), SOURCE_PATH, "exec"), namespace)
    return namespace[name]


def test_hook_is_registered_for_grpo():
    src = open(SOURCE_PATH, encoding = "utf-8").read()
    assert (
        'RL_CONFIG_CHANGES["grpo_trainer"].append(grpo_config_warn_rollout_logps_as_old_delta)'
        in src
    )


CONFIG_SOURCE = """
class GRPOConfig:
    loss_type: str = "dapo"
    delta: float | None = None
"""


def _config_check(monkeypatch, env, **config_locals):
    if env is None:
        monkeypatch.delenv("UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD", env)
    snippet = _load("grpo_config_warn_rollout_logps_as_old_delta")("", CONFIG_SOURCE)
    # Generated into UnslothGRPOConfig.__init__, whose arguments are locals.
    code = "def __init__(" + ", ".join(f"{k} = {v!r}" for k, v in config_locals.items()) + "):\n"
    code += "".join("    " + line + "\n" for line in snippet.splitlines())
    code += "    return None\n"
    namespace = {"os": os}
    exec(code, namespace)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        namespace["__init__"]()
    return buf.getvalue()


@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"])
def test_rollout_logps_as_old_without_delta_warns(monkeypatch, loss_type):
    out = _config_check(monkeypatch, "1", loss_type = loss_type, delta = None)
    assert "UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD=1 is set without `delta`" in out


@pytest.mark.parametrize(
    "env, loss_type, delta",
    [
        ("1", "dapo", 3.0),
        (None, "dapo", None),
        ("0", "dapo", None),
        ("1", "cispo", None),
        ("1", "sapo", None),
    ],
)
def test_no_warning_otherwise(monkeypatch, env, loss_type, delta):
    assert _config_check(monkeypatch, env, loss_type = loss_type, delta = delta) == ""


def test_config_without_delta_field_gets_no_check():
    assert (
        _load("grpo_config_warn_rollout_logps_as_old_delta")(
            "", "class GRPOConfig:\n    beta: float = 0.0\n"
        )
        == ""
    )
