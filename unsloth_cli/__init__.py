# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os as _os
import sys as _sys

# Entry-point-only behaviour (stream guard, -np<N> rewrite): must not reach a host that imports us.
_entry_base = _os.path.basename(_sys.argv[0]).lower() if _sys.argv else ""
_is_entry_point = _entry_base in {"unsloth", "unsloth.exe"}
_windows_studio_mutation_entry = (
    _sys.platform == "win32"
    and (_is_entry_point or _entry_base == "-m")
    and len(_sys.argv) >= 3
    and _sys.argv[1] == "studio"
    and _sys.argv[2] in {"setup", "update"}
)


_streams_reconfigured = False


def _reconfigure_entry_point_streams():
    """Give the console script streams that can render typer's help.

    Typer renders help via rich, whose box characters cp1252 and cp437 cannot encode,
    so `unsloth --help` dies once stdout is a pipe or a file. Windows gets UTF-8, as
    unsloth/__init__ already does; elsewhere the caller's encoding is kept and only
    the error handler is relaxed, so an explicit PYTHONIOENCODING still picks the
    bytes and only loses unencodable glyphs.

    Called at most once per process. The console script reaches it twice, from
    the import-time gate and again through _prepare_entry_point, and off Windows
    the second call did repeat the work: passing encoding = None keeps the
    caller's encoding, so the "already utf" guard below cannot become true and a
    C-locale console was reconfigured, and flushed, one time more than the
    console script ever did before this file grew a second entry route.
    """
    global _streams_reconfigured
    if _streams_reconfigured:
        return
    _streams_reconfigured = True
    _to_utf8 = _sys.platform == "win32"
    for _name in ("stdout", "stderr"):
        _stream = getattr(_sys, _name, None)
        try:
            if "utf" not in (_stream.encoding or "").lower():
                _stream.reconfigure(encoding = "utf-8" if _to_utf8 else None, errors = "replace")
        except Exception:
            pass


# Before typer, which binds the stream.
if _is_entry_point:
    _reconfigure_entry_point_streams()

# Before typer and the command imports: notebooks run this right after `pip install --no-deps unsloth`.
# `-m` alone could be a host package importing us.
_orig_argv = getattr(_sys, "orig_argv", [])
_runs_unsloth_cli = "-munsloth_cli" in _orig_argv or any(
    a == "-m" and b == "unsloth_cli" for a, b in zip(_orig_argv, _orig_argv[1:])
)
if (_is_entry_point or (_entry_base == "-m" and _runs_unsloth_cli)) and _sys.argv[1:2] == [
    "install-kernels"
]:
    from unsloth_cli._install_kernels import main as _install_kernels_main
    from unsloth_cli._ssl_keylog import drop_unwritable_ssl_keylog_file

    # The availability probe and the installers open HTTPS clients.
    drop_unwritable_ssl_keylog_file()
    _sys.exit(_install_kernels_main(_sys.argv[2:]))

from unsloth_cli._system_dir_guard import check_working_directory as _check_working_directory

# Move out of System32 before command imports: STUDIO_HOME resolves at import, and a relative UNSLOTH_STUDIO_HOME would hit System32.
_startup_guard = (
    _check_working_directory(_sys.argv[1:], _os.environ, _sys.platform) if _is_entry_point else None
)

from unsloth_cli._ssl_keylog import (
    drop_unwritable_ssl_keylog_file as _drop_unwritable_ssl_keylog_file,
)

# After the move, so a relative path is judged where ssl opens it; before any HTTPS client, since backend workers inherit it.
if _is_entry_point:
    _drop_unwritable_ssl_keylog_file()

import typer
from collections.abc import MutableMapping as _MutableMapping
from importlib.metadata import version as package_version, PackageNotFoundError


from unsloth_cli._argv import _expand_attached_np_short

# typer lists commands before sub-apps, each in registration order.
_COMMAND_ORDER = (
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
)

# Only a real dispatch loads command modules on first use; help, version, completion and library
# imports stay eager so Desktop's `unsloth -h` probe still fails when any command module is broken.
_EAGER_TOKENS = frozenset(
    ("-h", "--help", "-V", "--version", "--install-completion", "--show-completion")
)
_cli_args = (
    _sys.argv[1:] if (_is_entry_point or (_entry_base == "-m" and _runs_unsloth_cli)) else None
)
_load_every_command = not _windows_studio_mutation_entry and (
    not _cli_args
    or _cli_args[0] not in _COMMAND_ORDER
    or any(_arg in _EAGER_TOKENS for _arg in _cli_args)
    or "_UNSLOTH_COMPLETE" in _os.environ
)
del _cli_args

if _load_every_command:
    from unsloth_cli.commands.train import train
    from unsloth_cli.commands.inference import inference
    from unsloth_cli.commands.chat import chat
    from unsloth_cli.commands.start import start_app
    from unsloth_cli.commands.export import export, list_checkpoints
    from unsloth_cli.commands.eval import evaluate as eval_command
    from unsloth_cli.commands.doctor import doctor
    from unsloth_cli.commands.mcp import mcp_app
    from unsloth_cli.commands.studio import run as studio_run, _LlamaPassthroughCommand

# Always eager: `unsloth studio run` is the command Desktop starts the backend with.
from unsloth_cli.commands.studio import studio_app


_entry_point_prepared = False


def _prepare_entry_point():
    """Apply the `unsloth` console-script behaviour to this process.

    Split out for `python -m unsloth_cli`, which cannot use the argv[0] check
    above: `-m` imports this package in order to find unsloth_cli/__main__.py,
    so __init__ runs while sys.argv[0] is still "-m" and the gate cannot fire.
    __main__ rewrites argv[0] and calls this instead.

    Idempotent, because the console script reaches it through the gate below
    and only the module entry calls it by hand.
    """
    global _entry_point_prepared
    if _entry_point_prepared:
        return
    _reconfigure_entry_point_streams()
    _drop_unwritable_ssl_keylog_file()
    _expand_attached_np_short()
    # Set last, so a raise leaves the work retryable rather than silently skipped.
    _entry_point_prepared = True


# Canonicalise `-np<N>` only under the console-script; imports keep their argv intact.
if _is_entry_point:
    _prepare_entry_point()
del _entry_base, _is_entry_point, _orig_argv, _runs_unsloth_cli


def show_version(value: bool):
    if value:
        try:
            version = package_version("unsloth")
        except PackageNotFoundError:
            version = "unknown"
        typer.echo(f"unsloth {version}")
        raise typer.Exit()


_ARGV_META_KEY = "unsloth.invocation_args"

try:
    from typer.core import TyperGroup as _TyperGroup
except Exception:  # pragma: no cover - a typer without the public group class
    _TyperGroup = None

if _TyperGroup is not None:

    class _ArgvCapturingGroup(_TyperGroup):
        """Remember the tokens this invocation was given.

        Click hands the group its full argument list here and then keeps the tail
        on the child context, out of the callback's reach. Both `app(args = [...])`
        and CliRunner reach this, so a library call is classified by its own
        arguments rather than by the host's argv.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.commands = _LazyCommands(self.commands, _ACTIVE_LAZY_COMMANDS)
            if _load_every_command:
                # A command typer cannot build must still fail `unsloth -h` and `--version`.
                for _name in list(self.commands):
                    self.commands[_name]

        def parse_args(self, ctx, args):
            ctx.meta.setdefault(_ARGV_META_KEY, list(args))
            return super().parse_args(ctx, args)

else:  # pragma: no cover
    _ArgvCapturingGroup = None


app = typer.Typer(
    help = "Command-line interface for Unsloth training, inference, and export.",
    context_settings = {"help_option_names": ["-h", "--help"]},
    **({"cls": _ArgvCapturingGroup} if _ArgvCapturingGroup is not None else {}),
)


def _invocation_args(ctx):
    """The arguments this invocation was given, not the host process's argv.

    A library calling `app(args = [...])` or CliRunner never touches sys.argv, so
    reading it there would classify somebody else's command line and could move
    the process out from under the caller's relative paths. The `unsloth` console
    script never reaches here: it is classified at import, from the real argv.
    """
    captured = ctx.meta.get(_ARGV_META_KEY)
    if captured is not None:
        return list(captured)
    if not ctx.invoked_subcommand:
        return _sys.argv[1:]
    # No capture and no tail: assume it holds a path, so refuse rather than relocate.
    return [ctx.invoked_subcommand, *(list(getattr(ctx, "args", None) or []) or ["..."])]


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        callback = show_version,
        is_eager = True,
        help = "Show version and exit.",
    ),
):
    # Consume the import-time result once: a host can chdir between repeated app() calls.
    global _startup_guard
    _guard, _startup_guard = _startup_guard, None
    if _guard is None:
        # A host reaches this after commands.studio cached STUDIO_HOME, so moving now strands that root.
        _guard = _check_working_directory(
            _invocation_args(ctx),
            _os.environ,
            _sys.platform,
            relocate = False,
        )
    _message, _colour, _fatal = _guard
    if _message is not None:
        typer.secho(_message, fg = _colour, err = True)
    if _fatal:
        raise typer.Exit(code = 1)


_LAZY_COMMANDS = (
    "train",
    "inference",
    "chat",
    "export",
    "eval",
    "list-checkpoints",
    "doctor",
    "run",
    "start",
    "mcp",
    "connect",
)
_ACTIVE_LAZY_COMMANDS = () if _windows_studio_mutation_entry else _LAZY_COMMANDS


def _register_command(app, name):
    if name == "train":
        from unsloth_cli.commands.train import train
        app.command()(train)
    elif name == "inference":
        from unsloth_cli.commands.inference import inference
        app.command()(inference)
    elif name == "chat":
        from unsloth_cli.commands.chat import chat
        app.command()(chat)
    elif name == "export":
        from unsloth_cli.commands.export import export
        app.command()(export)
    elif name == "eval":
        from unsloth_cli.commands.eval import evaluate as eval_command
        app.command("eval")(eval_command)
    elif name == "list-checkpoints":
        from unsloth_cli.commands.export import list_checkpoints
        app.command("list-checkpoints")(list_checkpoints)
    elif name == "doctor":
        from unsloth_cli.commands.doctor import doctor
        app.command("doctor")(doctor)
    elif name == "start":
        from unsloth_cli.commands.start import start_app
        app.add_typer(
            start_app,
            name = "start",
            help = "Start a coding agent (Claude, Codex, OpenClaw, OpenCode, Hermes, Pi, dsh, Vibe, "
            "VS Code) against Unsloth.",
        )
    elif name == "mcp":
        from unsloth_cli.commands.mcp import mcp_app
        app.add_typer(
            mcp_app, name = "mcp", help = "Connect coding agents to the Unsloth Studio MCP server."
        )
    elif name == "connect":
        from unsloth_cli.commands.start import start_app

        # backwards-compatible hidden alias: `unsloth connect` routes to `unsloth start`.
        app.add_typer(
            start_app,
            name = "connect",
            hidden = True,
            help = "Deprecated alias for `unsloth start`.",
        )
    elif name == "run":
        from unsloth_cli.commands.studio import run as studio_run, _LlamaPassthroughCommand

        # top-level `unsloth run` aliases `unsloth studio run`; same context so unknown flags pass through to llama-server.
        app.command(
            "run",
            cls = _LlamaPassthroughCommand,
            context_settings = {
                "allow_extra_args": True,
                "ignore_unknown_options": True,
            },
            help = "Alias for `unsloth studio run`.",
        )(studio_run)
    else:
        raise KeyError(name)


def _load_command(name):
    scratch = typer.Typer(add_completion = False)
    _register_command(scratch, name)
    return typer.main.get_group(scratch).commands[name]


_PENDING = object()


class _LazyCommands(_MutableMapping):
    """`group.commands` whose pending entries build on first lookup."""

    def __init__(self, eager, lazy_names):
        self._commands = {}
        for name in _COMMAND_ORDER:
            if name in eager:
                self._commands[name] = eager[name]
            elif name in lazy_names:
                self._commands[name] = _PENDING
        for name, command in eager.items():
            self._commands.setdefault(name, command)

    def pending(self, name):
        return self._commands.get(name) is _PENDING

    def __getitem__(self, name):
        command = self._commands[name]
        if command is _PENDING:
            command = self._commands[name] = _load_command(name)
        return command

    def __setitem__(self, name, command):
        self._commands[name] = command

    def __delitem__(self, name):
        del self._commands[name]

    def __contains__(self, name):
        return name in self._commands

    def __iter__(self):
        return iter(self._commands)

    def __len__(self):
        return len(self._commands)


app.add_typer(studio_app, name = "studio", help = "Unsloth Studio commands.")
if _ArgvCapturingGroup is None:  # pragma: no cover - no group class to defer through
    for _name in _COMMAND_ORDER:
        if _name in _ACTIVE_LAZY_COMMANDS:
            _register_command(app, _name)


# Names this module used to import eagerly, for callers that still read them here.
_LAZY_ATTRIBUTES = {
    "train": ("unsloth_cli.commands.train", "train"),
    "inference": ("unsloth_cli.commands.inference", "inference"),
    "chat": ("unsloth_cli.commands.chat", "chat"),
    "start_app": ("unsloth_cli.commands.start", "start_app"),
    "export": ("unsloth_cli.commands.export", "export"),
    "list_checkpoints": ("unsloth_cli.commands.export", "list_checkpoints"),
    "eval_command": ("unsloth_cli.commands.eval", "evaluate"),
    "doctor": ("unsloth_cli.commands.doctor", "doctor"),
    "mcp_app": ("unsloth_cli.commands.mcp", "mcp_app"),
    "studio_run": ("unsloth_cli.commands.studio", "run"),
    "studio_app": ("unsloth_cli.commands.studio", "studio_app"),
    "_LlamaPassthroughCommand": ("unsloth_cli.commands.studio", "_LlamaPassthroughCommand"),
}


def __getattr__(name):
    try:
        module_name, attribute = _LAZY_ATTRIBUTES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib
    return getattr(importlib.import_module(module_name), attribute)


@app.command(
    "install-kernels",
    context_settings = {
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
    help = "Install prebuilt xformers / flash-attn / causal_conv1d / mamba_ssm wheels matching the installed torch (all by default).",
)
def install_kernels(ctx: typer.Context):
    # Listed for `unsloth --help`; the console script dispatches before typer is imported.
    from unsloth_cli._install_kernels import main as _install_kernels_main
    raise typer.Exit(code = _install_kernels_main(ctx.args))
