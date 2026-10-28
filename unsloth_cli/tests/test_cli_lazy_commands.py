# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Top-level commands other than studio load on first use; `unsloth -h` answers from a table.

Desktop runs `unsloth -h` as its preflight and `unsloth studio run` to start the backend,
so neither may import train, the pydantic config, start, mcp, chat, inference, eval,
export or doctor. The table must render exactly what the loaded commands render.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import typer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import unsloth_cli  # noqa: E402

LAZY_MODULES = (
    "unsloth_cli.commands.train",
    "unsloth_cli.commands.inference",
    "unsloth_cli.commands.chat",
    "unsloth_cli.commands.export",
    "unsloth_cli.commands.eval",
    "unsloth_cli.commands.doctor",
    "unsloth_cli.commands.mcp",
    "unsloth_cli.commands.start",
    "unsloth_cli.config",
)


def _modules_after(args: list[str], *, stop_studio_run: bool = False) -> set[str]:
    """Run the console-script path in a fresh interpreter and report which watched modules loaded."""
    stop = ""
    if stop_studio_run:
        # Stop `studio run` at the first statement of its body, before any server work.
        stop = (
            "import unsloth_cli.commands.studio as _s\n"
            "def _stop(*a, **k):\n"
            "    raise SystemExit(0)\n"
            "_s._consume_start_api_key_marker_env = _stop\n"
        )
    probe = (
        "import json, sys\n"
        f"sys.argv = ['unsloth', *{args!r}]\n"
        "import unsloth_cli\n"
        f"{stop}"
        "try:\n"
        "    unsloth_cli.app()\n"
        "except SystemExit:\n"
        "    pass\n"
        f"watched = {(*LAZY_MODULES, 'pydantic')!r}\n"
        "print(json.dumps(sorted(m for m in watched if m in sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output = True,
        text = True,
        cwd = str(_REPO_ROOT),
        env = {**__import__("os").environ, "NO_COLOR": "1", "COLUMNS": "80"},
    )
    assert result.returncode == 0, result.stderr
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def _pydantic_preloaded() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import sys, typer, rich; print('pydantic' in sys.modules)"],
        capture_output = True,
        text = True,
    )
    return result.stdout.strip() == "True"


@pytest.mark.parametrize(
    "args",
    [["-h"], ["--help"], ["--version"], ["studio", "-h"], ["studio", "run", "--help"]],
)
def test_desktop_paths_load_no_other_command_module(args):
    loaded = _modules_after(args)
    if _pydantic_preloaded():
        loaded.discard("pydantic")
    assert loaded == set()


@pytest.mark.parametrize("args", [["studio", "run", "--model", "x"], ["run", "--model", "x"]])
def test_studio_run_body_starts_without_other_command_modules(args):
    loaded = _modules_after(args, stop_studio_run = True)
    if _pydantic_preloaded():
        loaded.discard("pydantic")
    assert loaded == set()


def test_a_named_command_loads_only_its_module():
    loaded = _modules_after(["list-checkpoints", "--help"])
    assert "unsloth_cli.commands.export" in loaded
    assert "unsloth_cli.commands.train" not in loaded
    assert "unsloth_cli.commands.start" not in loaded


def _render_help(group, width: int) -> str:
    ctx = group.make_context("unsloth", [], resilient_parsing = True, terminal_width = width)
    ctx.max_content_width = width
    return group.get_help(ctx)


@pytest.mark.parametrize("width", [40, 80, 200])
@pytest.mark.parametrize("rich_help", [True, False])
def test_table_renders_what_the_loaded_commands_render(monkeypatch, width, rich_help):
    monkeypatch.setenv("COLUMNS", str(width))
    if not rich_help:
        monkeypatch.setattr(typer.core, "HAS_RICH", False)
    group = typer.main.get_command(unsloth_cli.app)
    from_table = _render_help(group, width)
    assert any(group.commands.pending(name) for name in unsloth_cli._LAZY_COMMANDS)
    for name in list(group.commands):
        group.commands[name]
    assert not any(group.commands.pending(name) for name in group.commands)
    assert _render_help(group, width) == from_table


def test_table_rows_match_the_loaded_commands():
    group = typer.main.get_command(unsloth_cli.app)
    ctx = group.make_context("unsloth", [], resilient_parsing = True)
    group._listing += 1
    stubs = {name: group.get_command(ctx, name) for name in unsloth_cli._LAZY_COMMANDS}
    group._listing -= 1
    for name, stub in stubs.items():
        real = group.get_command(ctx, name)
        assert real is not stub
        assert real.name == stub.name == name
        assert real.help == stub.help, name
        assert real.short_help == stub.short_help, name
        assert bool(real.hidden) == bool(stub.hidden), name
        assert bool(real.deprecated) == bool(stub.deprecated), name
        assert not getattr(real, "rich_help_panel", None), name
        for limit in (10, 45, 80, 200):
            assert real.get_short_help_str(limit) == stub.get_short_help_str(limit), name


# What `unsloth -h` listed when every command module was imported up front.
EAGER_ORDER = [
    "train",
    "inference",
    "chat",
    "export",
    "eval",
    "list-checkpoints",
    "doctor",
    "run",
    "install-kernels",
    "studio",
    "start",
    "mcp",
    "connect",
]


def test_every_registered_command_has_a_fixed_position():
    """typer lists commands before sub-apps; a new one must be placed in _COMMAND_ORDER."""
    group = typer.main.get_command(unsloth_cli.app)
    names = list(group.commands)
    assert names == EAGER_ORDER
    assert set(names) <= set(unsloth_cli._COMMAND_ORDER)
    assert names == [n for n in unsloth_cli._COMMAND_ORDER if n in names]
    kinds = [isinstance(group.commands[n], typer.core.TyperGroup) for n in names]
    assert kinds == sorted(kinds), "commands must precede sub-apps, as typer orders them"


def test_completion_answers_from_the_table():
    group = typer.main.get_command(unsloth_cli.app)
    ctx = group.make_context("unsloth", [], resilient_parsing = True)
    before = [(c.value, c.help) for c in group.shell_complete(ctx, "")]
    assert any(group.commands.pending(name) for name in unsloth_cli._LAZY_COMMANDS)
    for name in list(group.commands):
        group.commands[name]
    assert [(c.value, c.help) for c in group.shell_complete(ctx, "")] == before


def test_former_module_attributes_still_resolve():
    from unsloth_cli.commands import eval as eval_mod, export, start, studio, train

    assert unsloth_cli.train is train.train
    assert unsloth_cli.eval_command is eval_mod.evaluate
    assert unsloth_cli.list_checkpoints is export.list_checkpoints
    assert unsloth_cli.start_app is start.start_app
    assert unsloth_cli.studio_run is studio.run
    assert unsloth_cli.studio_app is studio.studio_app
    assert unsloth_cli._expand_attached_np_short is studio._expand_attached_np_short
    with pytest.raises(AttributeError):
        unsloth_cli.not_a_command
