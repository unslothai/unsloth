# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os as _os
import sys as _sys

# Are we the `unsloth` console script, rather than a library import? Both the
# stream guard below and the `-np<N>` rewrite further down are entry-point
# behaviour and must not reach into a host application that imports us.
_entry_base = _os.path.basename(_sys.argv[0]).lower() if _sys.argv else ""
_is_entry_point = _entry_base in {"unsloth", "unsloth.exe"}


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

from unsloth_cli._system_dir_guard import check_working_directory as _check_working_directory

# Running from System32 or any subdir WILL cause errors if not prevented. A
# command the folder cannot affect (the ones Unsloth Desktop spawns, issue #8510)
# moves out of it; everything else stops in the callback below.
#
# Before the command imports, since unsloth_cli.commands.studio resolves
# STUDIO_HOME at import time and a relative UNSLOTH_STUDIO_HOME would otherwise
# be pinned to the folder we are leaving. The message waits for typer to render
# it. A library import reaches the same check from the callback instead.
_startup_guard = (
    _check_working_directory(_sys.argv[1:], _os.environ, _sys.platform) if _is_entry_point else None
)

import typer
from importlib.metadata import version as package_version, PackageNotFoundError


from unsloth_cli.commands.train import train
from unsloth_cli.commands.inference import inference
from unsloth_cli.commands.chat import chat
from unsloth_cli.commands.start import start_app
from unsloth_cli.commands.export import export, list_checkpoints
from unsloth_cli.commands.studio import (
    run as studio_run,
    studio_app,
    _expand_attached_np_short,
)


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
    _expand_attached_np_short()
    # Set last, so a raise leaves the work retryable rather than silently
    # skipped. Neither call can currently raise -- the first swallows everything
    # and the second is pure argv manipulation -- but the ordering costs nothing.
    _entry_point_prepared = True


# Canonicalise `-np<N>` only under the `unsloth` console-script;
# third-party scripts that import unsloth_cli keep their argv intact.
if _is_entry_point:
    _prepare_entry_point()
del _entry_base, _is_entry_point


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
    # No capture and no tail to read: assume it holds a path, so an invocation
    # that cannot be read in full is refused rather than relocated.
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
    # Consume the import-time result once: a host calling the app repeatedly can
    # chdir between calls, so each later call is checked afresh.
    global _startup_guard
    _guard, _startup_guard = _startup_guard, None
    if _guard is None:
        # A host reaches this after commands.studio has resolved STUDIO_HOME at
        # import time, so moving now would leave that cached root behind.
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


app.command()(train)
app.command()(inference)
app.command()(chat)
app.command()(export)
app.command("list-checkpoints")(list_checkpoints)
app.add_typer(studio_app, name = "studio", help = "Unsloth Studio commands.")
app.add_typer(
    start_app,
    name = "start",
    help = "Start a coding agent (Claude, Codex, OpenClaw, OpenCode, Hermes, Pi) against Unsloth.",
)
# Backwards-compatible hidden alias: `unsloth connect` routes to `unsloth start`.
app.add_typer(
    start_app,
    name = "connect",
    hidden = True,
    help = "Deprecated alias for `unsloth start`.",
)

# Top-level `unsloth run` aliases `unsloth studio run`; same context
# so unknown flags still pass through to llama-server.
app.command(
    "run",
    context_settings = {
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
    help = "Alias for `unsloth studio run`.",
)(studio_run)
