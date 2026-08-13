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


app = typer.Typer(
    help = "Command-line interface for Unsloth training, inference, and export.",
    context_settings = {"help_option_names": ["-h", "--help"]},
)


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        callback = show_version,
        is_eager = True,
        help = "Show version and exit.",
    ),
):
    if (
        _sys.platform == "win32"
    ):  # this block catches unsloth running inside of System32 or any subdirs, this WILL cause errors if not prevented.
        _windir = _os.environ.get("WINDIR", r"C:\Windows")
        # SysWOW64 too: a 32-bit elevated shell opens there, same unwritable folder.
        _system_dirs = [
            _os.path.normcase(_os.path.normpath(_os.path.join(_windir, _name)))
            for _name in ("System32", "SysWOW64")
        ]
        _cwd_raw = _os.getcwd()
        _cwd = _os.path.normcase(_os.path.normpath(_cwd_raw))
        if any(_cwd == _dir or _cwd.startswith(_dir + _os.sep) for _dir in _system_dirs):
            # Users land here via "Run as administrator", so name the folder and hand back
            # a runnable fix. SYSTEM's USERPROFILE is under System32 and would send them
            # straight back, so take the first home outside the Windows tree, or none.
            _windir_norm = _os.path.normcase(_os.path.normpath(_windir))
            _home = None
            for _candidate in (
                _os.environ.get("USERPROFILE"),
                _os.environ.get("PUBLIC"),
                _os.path.expanduser("~"),
            ):
                _norm = _os.path.normcase(_os.path.normpath(_candidate)) if _candidate else ""
                if _norm and _norm != _windir_norm and not _norm.startswith(_windir_norm + _os.sep):
                    _home = _candidate
                    break
            if _home:
                # Quote it, or C:\Users\Jane Doe reaches Set-Location as two arguments.
                # PowerShell single quotes are verbatim ('' escapes an apostrophe); cmd
                # needs double quotes once extensions are off. " is not legal in a path.
                _home_ps = "'" + _home.replace("'", "''") + "'"
                _home_cmd = '"' + _home + '"'
                _cd_lines = (
                    f"    cd {_home_ps}          (PowerShell)\n"
                    f"    cd /d {_home_cmd}       (cmd.exe)\n"
                )
            else:
                _cd_lines = f"    (any folder outside {_windir})\n"
            _argv = " ".join((f'"{_arg}"' if " " in _arg else _arg) for _arg in _sys.argv[1:])
            _retry = ("unsloth " + _argv).rstrip()
            typer.secho(
                f"Unsloth cannot run from {_cwd_raw}\n"
                "\n"
                "That is a Windows system folder. Unsloth writes caches, configs and model\n"
                "downloads into the current working directory, and Windows blocks that here.\n"
                "Opening a terminal with 'Run as administrator' starts you in a folder like\n"
                "this one, which is how most people end up here.\n"
                "\n"
                "Change to a normal folder and run the command again:\n"
                f"{_cd_lines}"
                f"    {_retry}",
                fg = "red",
                err = True,
            )
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
