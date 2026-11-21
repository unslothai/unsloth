# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import typer


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio_run():
    from unsloth_cli.commands.studio import studio_app
    return typer.main.get_command(studio_app).commands["run"]


def _top_level_run():
    from unsloth_cli import app
    return typer.main.get_command(app).commands["run"]


_COMMANDS = [pytest.param(_studio_run, id = "studio-run"), pytest.param(_top_level_run, id = "run")]


def _parse(get_command, args):
    command = get_command()
    defaults = command.make_context("run", []).params
    ctx = command.make_context("run", list(args))
    changed = {k: v for k, v in ctx.params.items() if defaults.get(k) != v}
    return list(ctx.args), changed


_LLAMA_SHORT_FLAGS = [
    ["-ctv", "q8_0"],
    ["-ctvd", "q8_0"],
    ["-dev", "CUDA0"],
    ["-devd", "CUDA1"],
    ["-mmdev", "CUDA0"],
    ["-kvo"],
    ["-nkvo"],
    ["-kvu"],
    ["-no-kvu"],
    ["-lv", "3"],
    ["-ctxcp", "8"],
    ["-sp"],
    ["-sps", "0.5"],
]


@pytest.mark.parametrize("get_command", _COMMANDS)
@pytest.mark.parametrize("extras", _LLAMA_SHORT_FLAGS, ids = lambda e: e[0])
def test_llama_short_flag_reaches_llama_server_whole(get_command, extras):
    args, changed = _parse(get_command, ["--model", "X", *extras])
    assert args == extras
    assert changed == {"model": "X"}


@pytest.mark.parametrize("get_command", _COMMANDS)
def test_mixed_tail_keeps_its_order(get_command):
    tail = ["-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on", "-dev", "CUDA0", "--rope-scaling", "yarn"]
    args, changed = _parse(get_command, ["--model", "X", *tail, "-p", "9001"])
    assert args == tail
    assert changed == {"model": "X", "port": 9001}


@pytest.mark.parametrize("get_command", _COMMANDS)
@pytest.mark.parametrize(
    "args,expected",
    [
        (["-p", "9001"], {"port": 9001}),
        (["-p9001"], {"port": 9001}),
        (["-v"], {"verbose": True}),
        (["-q"], {"silent": True}),
        (["-y"], {"yes": True}),
        (["-H", "0.0.0.0"], {"host": "0.0.0.0"}),
        (["-vq"], {"verbose": True, "silent": True}),
        (["-hf", "org/repo-GGUF"], {"model": "org/repo-GGUF"}),
        (["-np", "2"], {"parallel": 2}),
    ],
)
def test_run_own_short_options_unchanged(get_command, args, expected):
    rest, changed = _parse(get_command, args)
    assert rest == []
    assert changed == expected
