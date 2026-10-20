# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""All UV_* switch readers must follow uv's boolish rule: off, no, n and f are false, not true."""

from __future__ import annotations

import ast
import pathlib
import re
import subprocess

import pytest

from woa_ps_harness import (
    INSTALL_PS1,
    INSTALL_SRC,
    SETUP_PS1,
    SETUP_SRC,
    STACK_SRC,
    UV_POLICY_ENV,
    _function_source,
    _ps_copies,
    _ps_last,
    _script,
    clear_env,
    functions,
    requires_pwsh,
)


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[3]
SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"

# uv's set from crates/uv-static/src/lib.rs at 0.10.7, spelled out so it cannot follow the code.
UV_TRUE = ("1", "t", "true", "y", "yes", "on")
UV_FALSE = ("0", "f", "false", "n", "no", "off")

BOOLISH_TABLE = (
    ("1", True),
    ("true", True),
    ("yes", True),
    ("on", True),
    # The single letters. uv takes them, so a table that stopped at `true` would be wrong.
    ("t", True),
    ("y", True),
    ("TRUE", True),
    ("True", True),
    ("YES", True),
    ("ON", True),
    ("T", True),
    ("Y", True),
    # The four the old `-notin @("", "0", "false")` spelling got backwards.
    ("off", False),
    ("OFF", False),
    ("OfF", False),
    ("no", False),
    ("NO", False),
    ("n", False),
    ("N", False),
    ("f", False),
    ("F", False),
    ("0", False),
    ("false", False),
    ("FALSE", False),
    ("", False),
    ("   ", False),
    # uv exits on non-boolish values, so answer 'not set'; the old spelling read `2` as TRUE.
    ("2", False),
    ("01", False),
    ("maybe", False),
    ("tr", False),
    ("truee", False),
    ("offline", False),
    ("nope", False),
    # uv itself does not trim and aborts on these; every implementation must trim alike.
    ("  1  ", True),
    ("  true  ", True),
    ("\ttrue\t", True),
    ("  off  ", False),
    ("  0  ", False),
)

TABLE_IDS = [repr(value) for value, _ in BOOLISH_TABLE]


def _env(name: str, value: str | None) -> dict:
    """Set by subtraction: quotes, backticks or tabs in a value would not survive PowerShell source."""
    base = {"PATH": "/usr/bin:/bin"}
    for leaked in UV_POLICY_ENV:
        base.pop(leaked, None)
    if value is not None:
        base[name] = value
    return base


def _sh_offline(value: str | None, tmp_path: pathlib.Path) -> bool:
    """setup.sh's answer, from the real function body."""
    text = SETUP_SH.read_text(encoding = "utf-8")
    start = text.index("_uv_offline_requested() {")
    probe = tmp_path / "probe.sh"
    probe.write_text(
        text[start : text.index("\n}\n", start) + 3]
        + "\nif _uv_offline_requested; then echo yes; else echo no; fi\n"
    )
    # `sh` with a pinned PATH: on Windows `bash` is the WSL stub, which runs nothing.
    done = subprocess.run(
        ["sh", str(probe)],
        capture_output = True,
        text = True,
        env = _env("UV_OFFLINE", value),
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip() == "yes"


def _py_flag(function: str, name: str, value: str | None, monkeypatch) -> bool:
    """Lifts the function by AST, not import, since importing install_python_stack.py runs an installer."""
    node = next(
        (
            n
            for n in ast.parse(STACK_SRC).body
            if isinstance(n, ast.FunctionDef) and n.name == function
        ),
        None,
    )
    assert node is not None, f"{function} is gone from install_python_stack.py"
    import os

    namespace: dict = {"os": os}
    exec(compile(ast.Module(body = [node], type_ignores = []), "<stack>", "exec"), namespace)
    # A bare os.environ write outlives the test.
    monkeypatch.delenv(name, raising = False)
    if value is not None:
        monkeypatch.setenv(name, value)
    return bool(namespace[function](name))


def _others(name: str) -> tuple:
    """Every resolver variable except the one under test; clearing them all would delete the input."""
    return tuple(other for other in UV_POLICY_ENV if other != name)


def _ps_flag(script: pathlib.Path, function: str, name: str, value: str | None) -> bool:
    """One PowerShell script's answer, from the real function body."""
    src = INSTALL_SRC if script == INSTALL_PS1 else SETUP_SRC
    snippet = _script(
        clear_env(_others(name)),
        functions(src, function),
        f'Write-Output ([string]({function} "{name}"))',
    )
    return _ps_last(snippet, env = _env(name, value)) == "True"


@pytest.mark.parametrize(("value", "expected"), BOOLISH_TABLE, ids = TABLE_IDS)
def test_the_posix_shell_reads_uv_offline_from_uvs_table(value, expected, tmp_path):
    assert _sh_offline(value, tmp_path) is expected


@pytest.mark.parametrize(("value", "expected"), BOOLISH_TABLE, ids = TABLE_IDS)
def test_the_python_installer_reads_uv_offline_from_uvs_table(value, expected, monkeypatch):
    assert _py_flag("_uv_env_flag", "UV_OFFLINE", value, monkeypatch) is expected


@requires_pwsh
@pytest.mark.parametrize("script", [INSTALL_PS1, SETUP_PS1], ids = ["install.ps1", "setup.ps1"])
@pytest.mark.parametrize(("value", "expected"), BOOLISH_TABLE, ids = TABLE_IDS)
def test_powershell_reads_a_uv_flag_from_uvs_table(script, value, expected):
    assert _ps_flag(script, "Test-UvEnvFlag", "UV_OFFLINE", value) is expected


def test_an_unset_variable_is_not_a_request(tmp_path, monkeypatch):
    """The default, and the one row the table above cannot express."""
    assert _sh_offline(None, tmp_path) is False
    assert _py_flag("_uv_env_flag", "UV_OFFLINE", None, monkeypatch) is False


@requires_pwsh
@pytest.mark.parametrize("script", [INSTALL_PS1, SETUP_PS1], ids = ["install.ps1", "setup.ps1"])
def test_an_unset_variable_is_not_a_request_in_powershell(script):
    assert _ps_flag(script, "Test-UvEnvFlag", "UV_OFFLINE", None) is False


@requires_pwsh
@pytest.mark.parametrize("script", [INSTALL_PS1, SETUP_PS1], ids = ["install.ps1", "setup.ps1"])
@pytest.mark.parametrize("function", ["Test-UvEnvFlag", "Test-PipEnvFlag"])
def test_an_unset_variable_answers_rather_than_aborts_under_a_callers_strict_mode(script, function):
    """The same row as above, in the session setup.ps1 actually runs in.

    `(Get-Item "Env:$Name").Value` on an unset variable reads a property off `$null`, and
    under `Set-StrictMode -Version 2` or `Latest` that is a TERMINATING
    PropertyNotFoundException rather than an empty string. setup.ps1 never turns strict
    mode off -- four of its comments say they are written for a caller's Set-StrictMode --
    and unset is the ordinary state of UV_OFFLINE, so an operator whose profile sets strict
    mode had setup abort before installing anything. `[Environment]::GetEnvironmentVariable`
    returns `$null` for a missing variable instead, which the `[string]` cast makes "".

    Strict mode is set in the snippet, not asked of the runner, so this states the
    requirement on every host rather than only on one configured to reproduce it.
    """
    snippet = _script(
        "Set-StrictMode -Version Latest",
        clear_env(_others("UV_OFFLINE")),
        functions(INSTALL_SRC if script == INSTALL_PS1 else SETUP_SRC, function),
        f'Write-Output ([string]({function} "UV_OFFLINE"))',
    )
    assert _ps_last(snippet, env = _env("UV_OFFLINE", None)) == "False"


@requires_pwsh
@pytest.mark.parametrize("variable", ["UV_OFFLINE", "UV_NO_CONFIG", "UV_NO_INDEX"])
@pytest.mark.parametrize(("value", "expected"), BOOLISH_TABLE, ids = TABLE_IDS)
def test_every_uv_variable_gets_the_same_table(variable, value, expected):
    """One function, so the variable cannot change the answer.

    Asserted rather than assumed: the bug being fixed was three sites reading three
    variables through three inline copies of one idiom, and a later site added the same
    way would read differently again.
    """
    assert _ps_flag(SETUP_PS1, "Test-UvEnvFlag", variable, value) is expected


@pytest.mark.parametrize(("value", "expected"), BOOLISH_TABLE, ids = TABLE_IDS)
def test_the_python_installer_reads_pip_no_index_from_pips_table(value, expected, monkeypatch):
    assert _py_flag("_pip_env_flag", "PIP_NO_INDEX", value, monkeypatch) is expected


@requires_pwsh
@pytest.mark.parametrize("script", [INSTALL_PS1, SETUP_PS1], ids = ["install.ps1", "setup.ps1"])
@pytest.mark.parametrize(("value", "expected"), BOOLISH_TABLE, ids = TABLE_IDS)
def test_powershell_reads_pip_no_index_from_pips_table(script, value, expected):
    assert _ps_flag(script, "Test-PipEnvFlag", "PIP_NO_INDEX", value) is expected


def test_pips_rule_is_checked_against_pip_itself():
    """pip's own parser, on pip's own literals, so the table is not just our reading of it.

    Skipped rather than approximated where pip is not importable: an approximation here is
    the failure mode this test exists to rule out.
    """
    strtobool = pytest.importorskip("pip._internal.utils.misc").strtobool
    for value in UV_TRUE:
        assert strtobool(value) == 1, value
        assert strtobool(value.upper()) == 1, value
    for value in UV_FALSE:
        assert strtobool(value) == 0, value
        assert strtobool(value.upper()) == 0, value
    # Everything else is an error to pip, so the installers answer 'not set'.
    for value in ("", "   ", "2", "01", "maybe", "tr", "offline"):
        with pytest.raises(ValueError):
            strtobool(value)


def test_the_pip_reader_is_a_separate_function_from_the_uv_one():
    """Not a style point. PIP_NO_INDEX is pip's variable and uv never reads it, so a single
    shared function would mean a correction to uv's parsing silently changing what the pip
    fallback does with an index."""
    for source, uv_name, pip_name in (
        (INSTALL_SRC, "Test-UvEnvFlag", "Test-PipEnvFlag"),
        (SETUP_SRC, "Test-UvEnvFlag", "Test-PipEnvFlag"),
        (STACK_SRC, "_uv_env_flag", "_pip_env_flag"),
    ):
        assert uv_name in source and pip_name in source, (uv_name, pip_name)
    # And the pip one does not simply call the uv one.
    for source, uv_name, pip_name in (
        (INSTALL_SRC, "Test-UvEnvFlag", "Test-PipEnvFlag"),
        (SETUP_SRC, "Test-UvEnvFlag", "Test-PipEnvFlag"),
    ):
        assert uv_name not in _function_source(source, pip_name)


PS1_FILES = sorted(
    path
    for path in PACKAGE_ROOT.rglob("*.ps1")
    if ".git" not in path.parts and "tests" not in path.parts
)

# Matched as text so a new site written the old way fails.
OLD_IDIOM = '-notin @("", "0", "false")'

# (Get-Item Env:X).Value throws PropertyNotFoundException under Set-StrictMode 2+ when X is unset;
# -ErrorAction SilentlyContinue does not cover the property access.
_STRICT_UNSAFE_ENV_READ = re.compile(r"\(\s*Get-Item\s+[\"']?Env:[^)]*\)\s*\.Value")


@pytest.mark.parametrize("path", PS1_FILES, ids = lambda p: p.name)
def test_no_shipped_powershell_decides_a_resolver_flag_the_old_way(path):
    text = path.read_text(encoding = "utf-8-sig")
    offending = [
        line for line in text.splitlines() if OLD_IDIOM in line and not line.strip().startswith("#")
    ]
    assert not offending, (
        f"{path.name} decides a flag with {OLD_IDIOM}, which reads off/no/n/f as set. "
        "Use Test-UvEnvFlag for a UV_* variable or Test-PipEnvFlag for a PIP_* one."
    )


@pytest.mark.parametrize("path", PS1_FILES, ids = lambda p: p.name)
def test_no_shipped_powershell_reads_an_environment_variable_unsafely(path):
    """Env reads in shipped PowerShell must be strict-mode safe: an unset variable aborts the install."""
    text = path.read_text(encoding = "utf-8-sig")
    offending = [
        line.strip()
        for line in text.splitlines()
        if _STRICT_UNSAFE_ENV_READ.search(line) and not line.strip().startswith("#")
    ]
    assert not offending, (
        f"{path.name} reads an environment variable through (Get-Item Env:...).Value, which "
        "throws under a caller's Set-StrictMode when the variable is unset:\n  "
        + "\n  ".join(offending)
        + "\nUse [Environment]::GetEnvironmentVariable(name), which returns $null instead."
    )


@pytest.mark.parametrize("path", PS1_FILES, ids = lambda p: p.name)
def test_every_resolver_variable_in_shipped_powershell_goes_through_a_reader(path):
    """Every UV_*/PIP_* switch read in shipped PowerShell must go through the one reader function."""
    text = path.read_text(encoding = "utf-8-sig")
    lines = text.splitlines()
    for number, line in enumerate(lines, start = 1):
        stripped = line.strip()
        if stripped.startswith("#") or "ToLowerInvariant()" not in stripped:
            continue
        for variable in ("UV_OFFLINE", "UV_NO_INDEX", "UV_NO_CONFIG", "PIP_NO_INDEX"):
            # The window is the statement, since the value and the test can be two lines.
            window = "\n".join(lines[max(0, number - 3) : number + 1])
            assert variable not in window, (
                f"{path.name}:{number} lowercases a value near {variable} instead of "
                "calling Test-UvEnvFlag / Test-PipEnvFlag"
            )


@requires_pwsh
@pytest.mark.parametrize("name", ["Test-UvEnvFlag", "Test-PipEnvFlag", "Test-NoIndexRequested"])
def test_the_two_powershell_copies_are_identical(name):
    """install.ps1 and setup.ps1 cannot share code, so the parity is the assertion."""
    install, setup = _ps_copies(name)
    assert install == setup


def test_uv_no_index_is_not_a_uv_environment_variable():
    """UV_NO_INDEX is our convention, not a uv variable; uv only has the --no-index flag."""
    for source, reader, marker in (
        (INSTALL_SRC, "Test-NoIndexRequested", "function Test-NoIndexRequested"),
        (SETUP_SRC, "Test-NoIndexRequested", "function Test-NoIndexRequested"),
        (STACK_SRC, "_no_index_requested", "def _no_index_requested("),
    ):
        assert reader in source, reader
        # PowerShell documents above the function, Python below the def: take both.
        at = source.index(marker)
        window = source[max(0, at - 1800) : at + 1800]
        # Strip markers and collapse whitespace so a reflow cannot break the assertion.
        prose = " ".join(line.lstrip().lstrip("#").strip() for line in window.splitlines())
        prose = " ".join(prose.split())
        assert "defines no such environment variable" in prose, reader
        assert (
            "--no-index" in prose
        ), f"{reader} must say that uv's --no-index is a command-line flag only"

    # And no OTHER site reads the raw variable for its truth: one convention, one place.
    for source, reader in ((INSTALL_SRC, "Test-UvEnvFlag"), (SETUP_SRC, "Test-UvEnvFlag")):
        hits = [
            line
            for line in source.splitlines()
            if 'Test-UvEnvFlag "UV_NO_INDEX"' in line and not line.strip().startswith("#")
        ]
        assert len(hits) == 1, hits


@requires_pwsh
@pytest.mark.parametrize("script", [INSTALL_PS1, SETUP_PS1], ids = ["install.ps1", "setup.ps1"])
@pytest.mark.parametrize(("value", "expected"), BOOLISH_TABLE, ids = TABLE_IDS)
def test_our_no_index_convention_uses_uvs_spelling_by_choice(script, value, expected):
    """uv would ignore this variable whatever we did with it, so the table is a decision:
    a caller sets it beside UV_OFFLINE and UV_NO_CONFIG and means it the same way."""
    src = INSTALL_SRC if script == INSTALL_PS1 else SETUP_SRC
    snippet = _script(
        clear_env(_others("UV_NO_INDEX")),
        functions(src, "Test-UvEnvFlag", "Test-NoIndexRequested"),
        "Write-Output ([string](Test-NoIndexRequested))",
    )
    assert (_ps_last(snippet, env = _env("UV_NO_INDEX", value)) == "True") is expected


def test_we_do_not_silently_translate_our_convention_into_a_uv_flag():
    """We shape our own arguments; passing --no-index to uv would strip the index from working resolves."""
    for source in (INSTALL_SRC, SETUP_SRC):
        for line in source.splitlines():
            if line.strip().startswith("#"):
                continue
            assert not ("--no-index" in line and "UV_NO_INDEX" in line), line


@requires_pwsh
@pytest.mark.parametrize("script", [INSTALL_PS1, SETUP_PS1], ids = ["install.ps1", "setup.ps1"])
@pytest.mark.parametrize("value", ["off", "no", "n", "f", "0", "false"])
def test_uv_no_config_set_to_a_false_value_does_not_suppress_the_uv_config(script, value, tmp_path):
    """The unsafe direction, and the reason this is worth changing at all.

    Reading UV_NO_CONFIG=off as "true" makes Get-WoaUvConfigIndexPolicy return its empty
    default, which is indistinguishable from "no configuration exists". uv, meanwhile,
    reads `off` as false and goes on to honour the file. So the installer resolved against
    public PyPI while uv resolved against the caller's corporate mirror: a wrong index, not
    a conservative one.
    """
    src = INSTALL_SRC if script == INSTALL_PS1 else SETUP_SRC
    project = tmp_path / "proj"
    project.mkdir()
    (project / "uv.toml").write_text(
        'index-url = "https://pypi.corp.test/simple"\n', encoding = "utf-8"
    )
    snippet = _script(
        clear_env(_others("UV_NO_CONFIG")),
        f"Set-Location -LiteralPath '{project}'",
        functions(
            src,
            "Test-UvEnvFlag",
            "Remove-WoaTomlComment",
            "Split-WoaTomlKey",
            "Read-WoaUvInlineIndexArray",
            "Read-WoaUvTomlIndexKeys",
            "Get-WoaUvConfigIndexPolicy",
        ),
        "Write-Output ([string](Get-WoaUvConfigIndexPolicy).DefaultIndex)",
    )
    found = _ps_last(snippet, env = _env("UV_NO_CONFIG", value))
    assert found == "https://pypi.corp.test/simple", (
        f"UV_NO_CONFIG={value!r} is false to uv, so the uv.toml must still be read; "
        f"got {found!r}"
    )


@requires_pwsh
@pytest.mark.parametrize("value", ["off", "no", "n", "f"])
def test_uv_offline_set_to_a_false_value_leaves_the_resolve_reaching_pypi(value):
    """The conservative direction, stated so the two are not confused.

    Misreading UV_OFFLINE=off as offline made Test-WoaResolveReachesPyPI answer "PyPI is
    unreachable", which KEEPS the bundled wheelhouse copy rather than deleting it. Nothing
    breaks; the user who asked to go online is quietly given the stale local wheel and no
    message says so. Still wrong, just not dangerous.
    """
    snippet = _script(
        clear_env(_others("UV_OFFLINE")),
        functions(
            INSTALL_SRC,
            "Test-UvEnvFlag",
            "Test-WoaUrlIsPublicPyPI",
            "Remove-WoaTomlComment",
            "Split-WoaTomlKey",
            "Read-WoaUvInlineIndexArray",
            "Read-WoaUvTomlIndexKeys",
            "Get-WoaUvConfigIndexPolicy",
            "Test-WoaResolveReachesPyPI",
        ),
        "Write-Output ([string](Test-WoaResolveReachesPyPI))",
    )
    assert _ps_last(snippet, env = _env("UV_OFFLINE", value)) == "True"


@requires_pwsh
@pytest.mark.parametrize("script", [INSTALL_PS1, SETUP_PS1], ids = ["install.ps1", "setup.ps1"])
@pytest.mark.parametrize(("variable", "resolver"), [("UV_NO_INDEX", "uv"), ("PIP_NO_INDEX", "pip")])
@pytest.mark.parametrize("value", ["off", "no", "n", "f"])
def test_a_no_index_set_to_a_false_value_still_names_an_index(script, variable, resolver, value):
    """Get-WoaDependencyIndexArgs returning @() means "name no index at all", which under
    a `--default-index` pin leaves the CUDA index as the only source for every shared
    dependency. A caller who wrote NO_INDEX=off asked for the opposite."""
    src = INSTALL_SRC if script == INSTALL_PS1 else SETUP_SRC
    snippet = _script(
        clear_env(_others(variable)),
        functions(
            src,
            "Test-UvEnvFlag",
            "Test-PipEnvFlag",
            "Remove-WoaTomlComment",
            "Split-WoaTomlKey",
            "Read-WoaUvInlineIndexArray",
            "Read-WoaUvTomlIndexKeys",
            "Get-WoaUvConfigIndexPolicy",
            "Get-WoaDependencyIndexArgs",
        ),
        f"Write-Output ('[' + ((Get-WoaDependencyIndexArgs -Resolver '{resolver}')"
        " -join '|') + ']')",
    )
    # UV_NO_CONFIG stays unset so the config walk runs as on a real host.
    got = _ps_last(snippet, env = _env(variable, value))
    assert got != "[]", f"{variable}={value!r} is false, so an index must still be named"
    assert "https://pypi.org/simple" in got, got
