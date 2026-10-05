# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One table for "is this UV_* variable set", in all four places that ask.

The installers read the caller's UV_* switches to decide whether uv will touch a network
and which indexes it will consult, and they do it in four separate languages:
setup.sh's ``_uv_offline_requested``, install_python_stack.py's ``_uv_env_flag``, and a
``Test-UvEnvFlag`` in each of install.ps1 and studio/setup.ps1. None of them can call
another -- install.ps1 is executed straight off the wire by ``irm | iex``, with no file and
no sibling module on disk -- so the only thing holding them together is this table.

The rule is uv's, not ours. uv 0.10.7 parses every boolish UV_* variable in
``crates/uv-static/src/lib.rs::parse_boolish_environment_variable``, which restates clap's
``str_to_bool``: true is ``y yes t true on 1``, false is ``n no f false off 0``,
case-insensitive, and anything else is a hard error rather than a guess.

The PowerShell half used to spell the question inline as
``-notin @("", "0", "false")``, which reads ``off``, ``no``, ``n`` and ``f`` as TRUE. Those
four are exactly the values uv reads as FALSE, so a caller who wrote ``UV_OFFLINE=off``
meaning "go online" got the offline path, and one who wrote ``UV_NO_CONFIG=off`` had their
uv.toml suppressed and resolved against the wrong index policy.

PIP_NO_INDEX is pip's variable and uv never reads it, so it gets its own function and its
own row set, taken from pip's ``strtobool`` (``pip/_internal/utils/misc.py``) via
``ConfigOptionParser._update_defaults``. The literals happen to coincide with uv's today;
they are asserted separately anyway, so that a divergence upstream shows up as a failure
here rather than as one resolver quietly adopting the other's rule.
"""

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

# uv's own set, from crates/uv-static/src/lib.rs at 0.10.7. Spelled out rather than
# derived, because a derived table would follow the implementation wherever it went.
UV_TRUE = ("1", "t", "true", "y", "yes", "on")
UV_FALSE = ("0", "f", "false", "n", "no", "off")

# (value, is the flag set). Every row is a value a caller could really write.
BOOLISH_TABLE = (
    # The ordinary spellings.
    ("1", True),
    ("true", True),
    ("yes", True),
    ("on", True),
    # The single letters. uv takes them, so a table that stopped at `true` would be wrong.
    ("t", True),
    ("y", True),
    # Case is not significant to uv, nor to pip.
    ("TRUE", True),
    ("True", True),
    ("YES", True),
    ("ON", True),
    ("T", True),
    ("Y", True),
    # The four that the old `-notin @("", "0", "false")` spelling got backwards. These are
    # the whole reason this file exists.
    ("off", False),
    ("OFF", False),
    ("OfF", False),
    ("no", False),
    ("NO", False),
    ("n", False),
    ("N", False),
    ("f", False),
    ("F", False),
    # The two it already got right.
    ("0", False),
    ("false", False),
    ("FALSE", False),
    # Unset, or set to nothing, is not a request for anything.
    ("", False),
    ("   ", False),
    # Not boolish at all. uv exits on these rather than resolving, so the only answer that
    # does not act on a value uv rejected is "not set". `2` is the one the old spelling
    # read as TRUE.
    ("2", False),
    ("01", False),
    ("maybe", False),
    ("tr", False),
    ("truee", False),
    ("offline", False),
    ("nope", False),
    # Padded. uv itself does not trim and aborts on all four, so the resolve fails whatever
    # is answered here; every implementation trims, and they must trim alike.
    ("  1  ", True),
    ("  true  ", True),
    ("\ttrue\t", True),
    ("  off  ", False),
    ("  0  ", False),
)

TABLE_IDS = [repr(value) for value, _ in BOOLISH_TABLE]


def _env(name: str, value: str | None) -> dict:
    """A clean resolver environment with one variable set, for a child process.

    Built by subtraction rather than by assignment inside the snippet: a value with a
    quote, a backtick or a tab in it does not survive being pasted into PowerShell source,
    and those are exactly the rows worth having.
    """
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
    # `sh` with a pinned POSIX PATH, not `bash`: on a Windows runner `bash` resolves to the
    # WSL stub, which answers in UTF-16 and runs nothing.
    done = subprocess.run(
        ["sh", str(probe)],
        capture_output = True,
        text = True,
        env = _env("UV_OFFLINE", value),
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip() == "yes"


def _py_flag(function: str, name: str, value: str | None, monkeypatch) -> bool:
    """install_python_stack.py's answer, executed out of the source text.

    The module is not imported: importing it runs an installer. The one function is lifted
    by AST instead, which also means a test that keeps passing after the function is
    renamed is impossible.
    """
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
    # monkeypatch, never a bare os.environ write: a bare write outlives the test and leaves
    # a resolver policy set for every other module in this directory.
    monkeypatch.delenv(name, raising = False)
    if value is not None:
        monkeypatch.setenv(name, value)
    return bool(namespace[function](name))


def _others(name: str) -> tuple:
    """Every resolver variable except the one under test.

    `clear_env(UV_POLICY_ENV)` is the right idiom for a lifted block and the wrong one
    here: the variable being measured is in that list, so clearing it inside the snippet
    would delete the input and leave every "true" row answering false.
    """
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


# ── uv's table, in all four languages ────────────────────────────────────────────────────


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


# ── pip's table, kept its own ────────────────────────────────────────────────────────────


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
    # Everything else is an error to pip, which is why the installers answer "not set"
    # rather than picking a side: pip would never have run with that value anyway.
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


# ── the sites ────────────────────────────────────────────────────────────────────────────


PS1_FILES = sorted(
    path
    for path in PACKAGE_ROOT.rglob("*.ps1")
    if ".git" not in path.parts and "tests" not in path.parts
)

# The spelling that was wrong. Matched as text so a new site written the old way fails
# here, which is the only thing that keeps four separated copies honest over time.
OLD_IDIOM = '-notin @("", "0", "false")'

#: Reading an environment variable through the provider and then taking `.Value`. Fine
#: while the variable is set; under a caller's `Set-StrictMode -Version 2` or `Latest` it
#: is a terminating PropertyNotFoundException the moment it is not, because the property
#: is being read off `$null`. `-ErrorAction SilentlyContinue` does not help: it suppresses
#: Get-Item's own error, not the property access on what it did not return.
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
    """Structural, because the behavioural rows above can only reach the two readers.

    Eighteen sites read a variable this way and every one of them is on the Windows on ARM
    resolver path, where the variables are normally unset, so a caller's strict mode turned
    an ordinary install into an abort at whichever site ran first. Fixing the two flag
    readers alone would have moved the abort one line down, to UV_CONFIG_FILE.
    """
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
    """A UV_*/PIP_* switch read for its truth, anywhere, must be read by the one function.

    Greps for the variable names rather than for the idiom: the previous bug was not a
    typo, it was five sites each deciding the question for themselves, and only a rule
    about the variables catches the sixth.
    """
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


# ── UV_NO_INDEX is ours, and the code must say so ────────────────────────────────────────


def test_uv_no_index_is_not_a_uv_environment_variable():
    """Recorded as an assertion because the code reads a UV_-prefixed name and a reader
    will otherwise assume uv defines it.

    uv 0.10.7 defines UV_OFFLINE and UV_NO_CONFIG as environment variables and does NOT
    define UV_NO_INDEX; `--no-index` exists only as a command-line flag. So our handling of
    UV_NO_INDEX is our own convention, and every site that reads it goes through a function
    whose name does not claim otherwise.
    """
    for source, reader, marker in (
        (INSTALL_SRC, "Test-NoIndexRequested", "function Test-NoIndexRequested"),
        (SETUP_SRC, "Test-NoIndexRequested", "function Test-NoIndexRequested"),
        (STACK_SRC, "_no_index_requested", "def _no_index_requested("),
    ):
        assert reader in source, reader
        # The documentation window: PowerShell puts it in the comment block ABOVE the
        # function, Python in the docstring below the def, so take both sides.
        at = source.index(marker)
        window = source[max(0, at - 1800) : at + 1800]
        # Comment markers stripped and whitespace collapsed before matching: the sentence
        # is wrapped across lines, and an assertion that a reflow can break is an assertion
        # that will be deleted rather than fixed.
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
    """Position taken and pinned: we shape the arguments we pass, we do not pass
    `--no-index` to uv. Turning UV_NO_INDEX into a real uv flag would make our behaviour and
    uv's agree, but it would also turn a resolve that works today into one with no index at
    all. If that is ever done deliberately, this test is the place it gets discussed."""
    for source in (INSTALL_SRC, SETUP_SRC):
        for line in source.splitlines():
            if line.strip().startswith("#"):
                continue
            assert not ("--no-index" in line and "UV_NO_INDEX" in line), line


# ── what the callers do with the answer ──────────────────────────────────────────────────


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
    # UV_NO_CONFIG is left unset here on purpose, so the config walk runs the way it does
    # on a real host; the assertion is only that an index was named.
    got = _ps_last(snippet, env = _env(variable, value))
    assert got != "[]", f"{variable}={value!r} is false, so an index must still be named"
    assert "https://pypi.org/simple" in got, got
