# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`python -m unsloth_cli` must be the console script, byte for byte.

Windows materialises the `unsloth` entry point as a generated, unsigned
`unsloth.exe`, and an Application Control policy (AppLocker / WDAC / Smart App
Control) denies it while the signed interpreter beside it keeps running, so the
installer, the desktop app and locked-down users all need a route to the CLI
that does not go through that executable (issue #8490).

Two such routes exist and both must behave exactly like the console script,
because everything above them assumes the swap is invisible:

  * ``python -X utf8 -m unsloth_cli`` -- the public, documented one.
  * ``python -X utf8 -c "<trampoline>"`` -- the internal one, used by
    install.ps1, studio/src-tauri and the `studio run` respawn. It is spelled
    out here rather than imported so a silent edit to the constant on either
    side of the language boundary fails this test.

`sys.argv[0] = 'unsloth'` is what buys that equivalence: unsloth_cli/__init__
gates its entry-point behaviour (UTF-8 streams, the `-np<N>` rewrite) on the
basename of argv[0], and typer/click derive the program name printed in every
usage and error string from it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Byte-identical to WINDOWS_CLI_ENTRYPOINT in studio/src-tauri/src/process.rs, $script:UnslothCliTrampoline in
# install.ps1, and _WINDOWS_CLI_ENTRYPOINT in unsloth_cli/commands/studio.py.
TRAMPOLINE = (
    "import sys, os; sys.path[:1] = [x for x in sys.path[:1] if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]; "
    "sys.argv[0] = 'unsloth'; from unsloth_cli import app; sys.exit(app())"
)

# No -I.
# It implies -E, which would discard every PYTHON* variable the console script honours, and that divergence is exactly
# what the sys.path[:1] filter in the trampoline exists to avoid needing.
INTERPRETER = [sys.executable, "-X", "utf8"]

# Not .resolve(): a POSIX venv's bin/python is a symlink to the base interpreter, and resolving it would look for the
# console script next to /usr/bin/python3.
_SCRIPT_DIR = Path(sys.executable).parent
_CONSOLE_SCRIPT = _SCRIPT_DIR / ("unsloth.exe" if os.name == "nt" else "unsloth")

requires_console_script = pytest.mark.skipif(
    not _CONSOLE_SCRIPT.is_file(),
    reason = f"no `unsloth` console script beside {sys.executable}; install the package first",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_PACKAGE = _REPO_ROOT / "unsloth_cli"


def _installed_package_dir() -> Path | None:
    """Where a child interpreter's `import unsloth_cli` actually lands.

    The trampoline strips the working directory from sys.path, so a child
    resolves the INSTALLED package and never this checkout by way of the cwd.
    When the two differ, the subprocess cases below would be testing a released
    wheel rather than the tree under test, so they skip instead of failing for
    the wrong reason. The in-process and source-contract cases still run
    everywhere.

    The probe therefore has to strip the cwd exactly as the trampoline does, or
    it would answer for a search path the tests never use.
    """
    probe = _run(
        [
            *INTERPRETER,
            "-c",
            "import sys, os; sys.path[:1] = [x for x in sys.path[:1] if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]; "
            "import unsloth_cli; print(os.path.dirname(unsloth_cli.__file__))",
        ]
    )
    if probe.returncode != 0:
        return None
    return Path(probe.stdout.decode("utf-8", "replace").strip())


def _run(argv: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run *argv* with captured bytes. Text mode would hide an encoding fault."""
    return subprocess.run(
        argv,
        capture_output = True,
        timeout = 120,
        env = env,
    )


_INSTALLED_PACKAGE = _installed_package_dir()

requires_this_checkout_installed = pytest.mark.skipif(
    _INSTALLED_PACKAGE is None or _INSTALLED_PACKAGE.resolve() != _REPO_PACKAGE,
    reason = (
        f"`import unsloth_cli` in a child resolves to {_INSTALLED_PACKAGE}, not "
        f"{_REPO_PACKAGE}; install this checkout (pip install -e .) to run the "
        "subprocess parity cases"
    ),
)


def _module_argv(*args: str) -> list[str]:
    return [*INTERPRETER, "-m", "unsloth_cli", *args]


def _trampoline_argv(*args: str) -> list[str]:
    return [*INTERPRETER, "-c", TRAMPOLINE, *args]


def _console_argv(*args: str) -> list[str]:
    return [str(_CONSOLE_SCRIPT), *args]


# Cover a clean exit, both help renderers (rich draws box characters here), and the two error
# shapes: an unknown option at the root and inside a subcommand.
PARITY_CASES = [
    pytest.param(["--version"], id = "version"),
    pytest.param(["--help"], id = "help"),
    pytest.param(["studio", "--help"], id = "studio-help"),
    pytest.param(["--definitely-not-a-flag"], id = "unknown-root-flag"),
    pytest.param(["studio", "run", "--definitely-not-a-flag"], id = "unknown-subcommand-flag"),
]


@requires_this_checkout_installed
def test_the_module_entry_point_exists():
    """A missing __main__.py degrades to a confusing "cannot be directly executed"."""
    result = _run(_module_argv("--version"))
    assert result.returncode == 0, (
        f"`python -m unsloth_cli --version` failed ({result.returncode}):\n"
        f"{result.stderr.decode('utf-8', 'replace')}"
    )
    assert result.stdout.startswith(b"unsloth "), result.stdout


@requires_console_script
@requires_this_checkout_installed
@pytest.mark.parametrize("args", PARITY_CASES)
def test_module_entry_matches_the_console_script(args):
    reference = _run(_console_argv(*args))
    module = _run(_module_argv(*args))
    assert module.returncode == reference.returncode
    assert module.stdout == reference.stdout
    assert module.stderr == reference.stderr


@requires_console_script
@requires_this_checkout_installed
@pytest.mark.parametrize("args", PARITY_CASES)
def test_trampoline_matches_the_console_script(args):
    """The form install.ps1 and the Tauri app use, pinned against the real thing."""
    reference = _run(_console_argv(*args))
    trampoline = _run(_trampoline_argv(*args))
    assert trampoline.returncode == reference.returncode
    assert trampoline.stdout == reference.stdout
    assert trampoline.stderr == reference.stderr


@requires_this_checkout_installed
@pytest.mark.parametrize(
    "argv_builder",
    [_module_argv, _trampoline_argv],
    ids = ["module", "trampoline"],
)
def test_the_program_name_is_unsloth_not_the_launcher(argv_builder):
    """Without the argv[0] rewrite, usage strings read `__main__.py` or `-c`."""
    result = _run(argv_builder("--help"))
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    text = result.stdout.decode("utf-8", "replace")
    assert "unsloth" in text
    assert "__main__.py" not in text
    # `-c` would surface as the program name in the Usage line.
    assert "Usage: -c" not in text


def test_the_attached_np_short_is_still_canonicalised(monkeypatch):
    """`-np8` must reach typer as `-np 8`, not click's `-n -p 8`.

    This is the one thing a naive __main__.py silently loses. The gate in
    unsloth_cli/__init__ keys on argv[0], and `-m` imports the package to find
    __main__, so the gate has already run and seen "-m" before __main__ can fix
    argv[0]. The damage is quiet and severe: click reads `-np8` as `-n -p 8` and
    `-p` is --port, so `unsloth studio run -np8` was observed serving on port 8
    instead of 8888 with the parallel count dropped.

    Driven in-process because the outward symptom is a bound socket: only a
    started server reveals the wrong port, and the argv the CLI is handed is the
    same fact one step earlier.
    """
    import runpy

    import unsloth_cli

    recorded = {}

    def fake_app(*args, **kwargs):
        recorded["argv"] = list(sys.argv)
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(unsloth_cli, "app", fake_app)
    monkeypatch.setattr(unsloth_cli, "_entry_point_prepared", False)
    monkeypatch.setattr(sys, "argv", ["-m", "studio", "run", "-np8"])

    # SystemExit, because __main__ ends in sys.exit(app()) exactly as the console script does.
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("unsloth_cli", run_name = "__main__", alter_sys = True)
    assert exit_info.value.code in (None, 0)

    assert recorded["argv"] == ["unsloth", "studio", "run", "-np", "8"], (
        "__main__ must apply the console-script argv canonicalisation; got " f"{recorded['argv']}"
    )
    # Without this click prints `Usage: python -m unsloth_cli`, because it reads __main__.__package__ rather than
    # argv[0].
    assert recorded["kwargs"].get("prog_name") == "unsloth"


@requires_console_script
@requires_this_checkout_installed
@pytest.mark.parametrize(
    "argv_builder",
    [_module_argv, _trampoline_argv],
    ids = ["module", "trampoline"],
)
def test_help_matches_the_console_script_under_a_narrow_encoding(argv_builder):
    """rich draws box characters cp1252 cannot encode; --help must still agree.

    -X utf8 is dropped for this case on purpose: it would hand the child a utf
    stream anyway, which is exactly the situation the stream guard in
    unsloth_cli/__init__ does not need to handle.
    """
    strip = ("-X", "utf8")
    argv = [arg for arg in argv_builder("--help") if arg not in strip]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "cp1252"

    reference = _run(_console_argv("--help"), env = env)
    result = _run(argv, env = env)

    assert result.returncode == reference.returncode, (
        "--help died under a narrow stdout encoding:\n"
        f"{result.stderr.decode('utf-8', 'replace')}"
    )
    assert result.stdout == reference.stdout
    assert result.stderr == reference.stderr


def test_the_module_entry_source_keeps_its_two_load_bearing_details():
    """Runs everywhere, including where the subprocess cases skip.

    The parity cases above need this checkout installed, so on a machine holding
    a released wheel they would go quiet and a deleted __main__.py or a dropped
    prog_name would sail through. Both details are invisible at a glance and
    each has already been shipped wrong once, so pin them in the source too.
    """
    source = (_REPO_PACKAGE / "__main__.py").read_text(encoding = "utf-8")

    argv_assignment = source.find('sys.argv[0] = "unsloth"')
    package_import = source.find("import unsloth_cli")
    assert argv_assignment != -1, "__main__.py no longer rewrites argv[0]"
    assert package_import != -1, "__main__.py no longer imports the package"
    assert argv_assignment < package_import, (
        "argv[0] must be rewritten before the package is imported, or a direct "
        "`python path/to/__main__.py` run misses the console-script gate"
    )
    # `-m` imports the package to locate this module, so __init__ has already run
    # with argv[0] == "-m" and its gate cannot fire; __main__ has to say so.
    assert "_prepare_entry_point()" in source
    # click reads __main__.__package__ rather than argv[0] and would otherwise print `Usage: python -m unsloth_cli` in
    # every usage and error string.
    assert 'prog_name = "unsloth"' in source
    # The generated console script is `sys.exit(app())`.
    assert "sys.exit(unsloth_cli.app(" in source


@requires_this_checkout_installed
def test_the_advertised_module_route_ignores_a_shadowing_directory(tmp_path):
    """`-m` resolves the package before __main__.py runs, so -I is load bearing.

    A shell sitting in a directory that has an `unsloth_cli` folder is not exotic:
    it is anyone standing in an unsloth checkout. Without -I that copy wins and the
    printed recovery command drives the wrong install, which nothing inside
    __main__.py can detect or undo.
    """
    shadow = tmp_path / "unsloth_cli"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("app = None\n", encoding = "utf-8")
    (shadow / "__main__.py").write_text("print('SHADOWED')\n", encoding = "utf-8")

    plain = subprocess.run(
        [sys.executable, "-m", "unsloth_cli", "--version"],
        capture_output = True,
        timeout = 120,
        cwd = tmp_path,
    )
    assert (
        b"SHADOWED" in plain.stdout
    ), "the shadowing fixture did not take effect, so the case below proves nothing"

    isolated = _run([sys.executable, "-X", "utf8", "-I", "-m", "unsloth_cli", "--version"])
    assert isolated.returncode == 0, isolated.stderr.decode("utf-8", "replace")
    assert isolated.stdout.startswith(b"unsloth "), isolated.stdout


def test_every_advertised_module_route_is_isolated():
    """Runs everywhere: the commands we print must not lose their -I.

    Source-contract, because they live in hint text rather than in code we can call,
    and a copy that drops the flag reintroduces the shadowing silently.

    Only the three that name the MANAGED interpreter. -I implies -s, so it hides a
    `pip install --user` install from itself; __main__.py's docstring documents that
    case and offers the -c bootstrap instead, so it is not held to this rule.
    """
    advertised = {
        "studio/backend/routes/auth.py",
        "studio/backend/run.py",
        "install.ps1",
    }
    for name in sorted(advertised):
        source = (_REPO_ROOT / name).read_text(encoding = "utf-8")
        for line in source.splitlines():
            if "-m unsloth_cli" not in line:
                continue
            # click prints its own `Usage: python -m unsloth_cli` when prog_name is missing; that is the symptom being
            # described, not a command we offer.
            if "Usage:" in line:
                continue
            assert "-I -m unsloth_cli" in line, f"{name}: unisolated module route: {line.strip()}"


def test_the_module_docstring_documents_the_user_site_exception():
    """-I implies -s, so the advertised form cannot see a --user install.

    Measured: with the package in the user site, `python -m unsloth_cli` runs and
    `python -I -m unsloth_cli` reports "No module named unsloth_cli". Anyone hitting
    that has a launcher under %APPDATA% -- exactly the user-writable location a
    default AppLocker policy denies -- so it is the population this route exists for.
    """
    source = (_REPO_PACKAGE / "__main__.py").read_text(encoding = "utf-8")
    assert "pip install --user" in source
    assert "-I implies -s" in source
    assert (
        "sys.path[:1] = [x for x in sys.path[:1] if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]"
        in source
    )


@requires_console_script
@requires_this_checkout_installed
def test_safe_path_leaves_an_explicit_pythonpath_alone(tmp_path):
    """Under -P / PYTHONSAFEPATH there is no implicit entry to strip.

    Python then puts the first PYTHONPATH entry at sys.path[0], and the console
    script honours it. A filter that removed index 0 regardless would import a
    different package than the console script for the same environment, which is
    the one thing this change is not allowed to do. Measured before the guard
    existed: the console script loaded the shadow, the trampoline did not.
    """
    shadow = tmp_path / "shadow"
    (shadow / "unsloth_cli").mkdir(parents = True)
    (shadow / "unsloth_cli" / "__init__.py").write_text(
        "raise SystemExit('SHADOWED')\n", encoding = "utf-8"
    )
    env = dict(os.environ)
    env["PYTHONSAFEPATH"] = "1"
    env["PYTHONPATH"] = str(shadow)

    reference = _run(_console_argv("--version"), env = env)
    trampoline = _run(_trampoline_argv("--version"), env = env)

    assert trampoline.returncode == reference.returncode
    assert trampoline.stdout == reference.stdout
    assert trampoline.stderr == reference.stderr


@requires_console_script
@requires_this_checkout_installed
def test_the_working_directory_is_still_stripped_without_safe_path(tmp_path):
    """The other half: the guard must not disarm the filter it guards."""
    shadow = tmp_path / "shadow"
    (shadow / "unsloth_cli").mkdir(parents = True)
    (shadow / "unsloth_cli" / "__init__.py").write_text(
        "raise SystemExit('SHADOWED')\n", encoding = "utf-8"
    )
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        _trampoline_argv("--version"),
        capture_output = True,
        timeout = 120,
        env = env,
        cwd = shadow,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert result.stdout.startswith(b"unsloth "), result.stdout


def test_the_stream_reconfigure_happens_once_per_process(monkeypatch):
    """The console script reaches it twice; the streams must only move once.

    Off Windows the guard inside cannot short-circuit, because encoding = None
    deliberately keeps the caller's encoding, so the second call reconfigured a
    C-locale console again and flushed it again. Harmless, but it is a difference
    from what the console script did before this file grew a second entry route.
    """
    import unsloth_cli

    calls = []

    class _Stream:
        encoding = "ascii"

        def reconfigure(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(unsloth_cli, "_streams_reconfigured", False)
    monkeypatch.setattr(unsloth_cli._sys, "stdout", _Stream())
    monkeypatch.setattr(unsloth_cli._sys, "stderr", _Stream())

    unsloth_cli._reconfigure_entry_point_streams()
    unsloth_cli._reconfigure_entry_point_streams()
    unsloth_cli._reconfigure_entry_point_streams()

    assert len(calls) == 2, f"expected one reconfigure per stream, got {calls}"
