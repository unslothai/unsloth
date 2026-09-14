# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Setup reports a torchcodec that installed but cannot load.

Its wheel is Python-side only: at import it dlopens FFmpeg's avcodec/avutil.
Where those are absent it still installs, reports a version, and satisfies the
torch/torchcodec matrix notebook_validator enforces, then fails at import.
`datasets` 4.x decodes audio only through it and reports that as "please install
'torchcodec'", naming a package that is already installed.
"""

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_STUDIO = Path(__file__).resolve().parents[2]
_SETUP_SH = _STUDIO / "setup.sh"
_SETUP_PS1 = _STUDIO / "setup.ps1"


def _shipped_sh_probe() -> str:
    """The probe body as setup.sh ships it. Read, not copied: a copy would keep
    passing while the installer that actually runs on a user's machine drifted."""
    text = _SETUP_SH.read_text(encoding = "utf-8")
    start = text.index("_TORCHCODEC_PROBE='") + len("_TORCHCODEC_PROBE='")
    return text[start : text.index("'", start)]


def _shipped_ps1_probe() -> str:
    text = _SETUP_PS1.read_text(encoding = "utf-8")
    opener = "$_torchcodecProbe = @'\n"
    start = text.index(opener) + len(opener)
    return text[start : text.index("\n'@", start)]


_PROBE = _shipped_sh_probe()


def _raising(exc: str) -> str:
    """Force torchcodec's import into one outcome, leaving every other import alone."""
    return textwrap.dedent(
        f"""
        import builtins
        _real = builtins.__import__
        def _fake(name, *a, **k):
            if name == 'torchcodec':
                raise {exc}
            return _real(name, *a, **k)
        builtins.__import__ = _fake
        """
    )


def _ffmpeg(present: bool) -> str:
    """Pin the FFmpeg answer. A runner that happens to have it installed would
    otherwise flip the two loader states this distinction rests on."""
    return textwrap.dedent(
        f"""
        import ctypes.util, os
        ctypes.util.find_library = lambda n: {'"/usr/lib/libavutil.so"' if present else 'None'}
        os.environ['PATH'] = ''
        """
    )


def _probe(preamble: str) -> str:
    """The state the installers would read: the sentinel line, not all of stdout."""
    out = subprocess.run(
        ["python", "-c", preamble + _PROBE],
        capture_output = True,
        text = True,
    )
    states = [
        line[len("TORCHCODEC=") :].strip()
        for line in out.stdout.splitlines()
        if line.startswith("TORCHCODEC=")
    ]
    return states[-1] if states else ""


def _step_line(text: str, opening: str) -> str:
    start = text.index(f'step "torchcodec" "{opening}')
    return text[start : text.index("\n", start)]


def test_a_missing_torchcodec_reports_absent():
    # Nothing to say: absent says nothing about FFmpeg, and the soundfile path stands.
    # `name` is set the way the import system sets it, since that is what the probe reads.
    assert _probe(_raising("ModuleNotFoundError('no torchcodec', name = 'torchcodec')")) == "absent"


def test_an_unloadable_torchcodec_is_distinguished_from_an_absent_one():
    # ModuleNotFoundError is a subclass of ImportError, so catching ImportError
    # first would collapse the two states this whole report rests on.
    loadable = _probe(
        "import sys, types; sys.modules['torchcodec'] = types.ModuleType('torchcodec')\n"
    )
    assert loadable == "ok"

    unloadable = _probe(
        _ffmpeg(False)
        + _raising(
            "RuntimeError('Could not load libtorchcodec. 1. FFmpeg is not properly installed')"
        )
    )
    assert unloadable == "ffmpeg"


def test_ffmpeg_advice_needs_ffmpeg_to_actually_be_missing():
    # The loader message names a missing FFmpeg, a torch mismatch and another native
    # dependency in one breath, so the text cannot pick between them. With FFmpeg
    # already on the loader path it is one of the others, and sending someone to
    # install FFmpeg would point them at the wrong thing.
    same_error = _raising(
        "RuntimeError('Could not load libtorchcodec. 1. FFmpeg is not properly installed')"
    )
    assert _probe(_ffmpeg(True) + same_error) == "native"
    assert _probe(_ffmpeg(False) + same_error) == "ffmpeg"


def test_a_missing_transitive_module_is_not_read_as_an_absent_package():
    # torchcodec is installed and importing it raises the same class an absent one
    # would. Reporting that as absent leaves a damaged install with no warning.
    assert _probe(_raising("ModuleNotFoundError('no numpy', name = 'numpy')")) == "broken"
    # A damaged wheel missing one of torchcodec's own submodules is the same story:
    # the package is there. Only the top-level name is absent, which is the only
    # name an actually-absent package can raise from `import torchcodec`.
    assert _probe(_raising("ModuleNotFoundError('gone', name = 'torchcodec.decoders')")) == "broken"


def test_the_state_is_read_as_a_line_not_as_all_of_stdout():
    # Importing torch can print to stdout, and a state read as the whole buffer then
    # matches none of the arms, so setup silently drops the report.
    noisy = "import sys; print('banner from a startup hook'); "
    assert (
        _probe(noisy + _raising("ModuleNotFoundError('no torchcodec', name = 'torchcodec')"))
        == "absent"
    )


def test_an_unrelated_import_failure_is_not_blamed_on_ffmpeg():
    # A damaged wheel or a torch/torchcodec ABI mismatch also raises here, and
    # telling someone to install FFmpeg would send them at the wrong thing.
    assert _probe(_raising("ImportError('DLL load failed while importing _core')")) == "broken"


@pytest.mark.parametrize("script", [_SETUP_SH, _SETUP_PS1], ids = ["sh", "ps1"])
def test_both_installers_report_the_ffmpeg_case(script):
    # Spelled out past "cannot load", since the non-FFmpeg loader failure opens the same way.
    line = _step_line(script.read_text(encoding = "utf-8"), "installed but cannot load its FFmpeg")
    # Names the real dependency, and what still works, like the whisper.cpp steps do.
    assert "FFmpeg" in line
    assert "soundfile" in line
    # The formats libsndfile actually covers, so nobody reads this as "audio is dead".
    assert "wav/flac/mp3/ogg" in line


@pytest.mark.parametrize("script", [_SETUP_SH, _SETUP_PS1], ids = ["sh", "ps1"])
def test_both_installers_keep_ffmpeg_advice_out_of_the_other_failure(script):
    line = _step_line(script.read_text(encoding = "utf-8"), "installed but fails to import")
    assert "install an FFmpeg" not in line, "the non-FFmpeg failure still sends them at FFmpeg"
    assert "reinstall torchcodec" in line


@pytest.mark.parametrize("script", [_SETUP_SH, _SETUP_PS1], ids = ["sh", "ps1"])
def test_both_installers_report_the_loader_failure_that_is_not_ffmpeg(script):
    line = _step_line(script.read_text(encoding = "utf-8"), "installed but cannot load its native")
    assert "install an FFmpeg" not in line, "FFmpeg is already there; this sends them at it anyway"
    # Both remaining causes, since nothing available here picks between them.
    assert "does not support" in line
    assert "torch" in line


@pytest.mark.parametrize("script", [_SETUP_SH, _SETUP_PS1], ids = ["sh", "ps1"])
def test_the_probe_is_skipped_when_python_deps_were_skipped(script):
    # Nothing is installed to probe, and the venv may not even exist.
    text = script.read_text(encoding = "utf-8")
    probe = text.index('step "torchcodec"')
    guard = max(
        text.rfind("_SKIP_PYTHON_DEPS", 0, probe),
        text.rfind("$SkipPythonDeps", 0, probe),
    )
    assert guard != -1, "the probe is not behind the skip-python-deps guard"


def test_the_shell_probe_is_skipped_in_llama_only_mode():
    # _SKIP_PYTHON_DEPS is assigned inside the base install, which llama-only skips
    # entirely, so under `set -u` a bare read here aborts the whole run. Both halves
    # matter: the gate keeps the slow probe off the update path, and the default
    # expansion keeps a later edit from reintroducing the unbound read.
    text = _SETUP_SH.read_text(encoding = "utf-8")
    probe = text.index("_TORCHCODEC_PROBE=")
    guard = text.rfind("if [ ", 0, probe)
    line = text[guard : text.index("\n", guard)]
    assert "_LLAMA_ONLY" in line
    assert "${_SKIP_PYTHON_DEPS:-" in line


def test_the_shell_probe_is_bounded():
    # Importing torchcodec imports torch; a wedged GPU runtime must not hang setup.
    text = _SETUP_SH.read_text(encoding = "utf-8")
    probe = text.index("_TORCHCODEC_PROBE=")
    after = text[probe : text.index('step "torchcodec"', probe)]
    assert "timeout 60 python -c" in after
    # ...and still runs where coreutils `timeout` is absent, as the GPU probes do.
    assert "command -v timeout" in after
    # That fallback path is the one stock macOS takes, so the deadline has to live in
    # the body too, exactly as the XPU probe in this same script carries it.
    assert "_alarm(60)" in _shipped_sh_probe()


@pytest.mark.parametrize("script", [_SETUP_SH, _SETUP_PS1], ids = ["sh", "ps1"])
def test_both_installers_read_the_state_as_a_line(script):
    # Anything on stdout ahead of the answer, a torch banner most likely, leaves a
    # whole-buffer read matching no arm at all, and the report vanishes.
    text = script.read_text(encoding = "utf-8")
    probe = text.index("TORCHCODEC=")
    after = text[probe : text.index('step "torchcodec"', probe)]
    assert "s/^TORCHCODEC=//p" in after or "(?m)^TORCHCODEC=" in after


def test_the_two_installers_ship_the_same_probe():
    # Separate copies drift silently, and then the two platforms report differently
    # for the same install. Quote style is the one allowed difference: setup.sh wraps
    # the body in single quotes so its Python strings must use double ones.
    def _same(body: str) -> str:
        return body.replace('"', "'").strip()

    assert _same(_shipped_sh_probe()) == _same(_shipped_ps1_probe())


def test_the_powershell_probe_is_bounded():
    text = _SETUP_PS1.read_text(encoding = "utf-8")
    probe = text.index("$_torchcodecProbe = ")
    after = text[probe : text.index('step "torchcodec"', probe)]
    assert "Invoke-BoundedPythonProbe" in after
    assert "-TimeoutSec" in after


def test_the_powershell_probe_carries_no_double_quote():
    # Invoke-BoundedPythonProbe wraps the body in double quotes to build -c <body>,
    # so one more anywhere in it (a comment included) closes that argument early.
    # Windows then splits the rest into stray argv entries and python runs the
    # truncated head, which still parses -- it exits 0 with no output and the whole
    # report goes quiet.
    assert '"' not in _shipped_ps1_probe()


# Runs the shipped helper against the shipped body, so neither can drift from what the
# test exercises. $args are setup.ps1, the interpreter, and a PYTHONPATH to import from.
_PWSH_HARNESS = """
$ps1 = (Get-Content -Raw $args[0]) -replace "`r`n", "`n"
$opener = "`$_torchcodecProbe = @'`n"
$start = $ps1.IndexOf($opener) + $opener.Length
$code = $ps1.Substring($start, $ps1.IndexOf("`n'@", $start) - $start)
$ast = [System.Management.Automation.Language.Parser]::ParseInput($ps1, [ref]$null, [ref]$null)
$fn = $ast.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $node.Name -eq 'Invoke-BoundedPythonProbe'
}, $true)[0]
Invoke-Expression $fn.Extent.Text
$env:PYTHONPATH = $args[2]
Write-Output (Invoke-BoundedPythonProbe -PythonExe $args[1] -Code $code -TimeoutSec 60).Output.Trim()
"""


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "pwsh unavailable")
def test_the_powershell_probe_survives_its_own_quoting(tmp_path):
    # Against the unloadable case on purpose. A body cut short at a double quote keeps
    # the ModuleNotFoundError branch, so a python that simply lacks torchcodec answers
    # `absent` either way and proves nothing; only this branch goes silent.
    (tmp_path / "torchcodec.py").write_text(
        "raise RuntimeError('Could not load libtorchcodec')", encoding = "utf-8"
    )
    harness = tmp_path / "probe_harness.ps1"
    harness.write_text(_PWSH_HARNESS, encoding = "utf-8")
    out = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-File",
            str(harness),
            str(_SETUP_PS1),
            sys.executable,
            str(tmp_path),
        ],
        capture_output = True,
        text = True,
    )
    assert out.returncode == 0, out.stderr
    assert "TORCHCODEC=ffmpeg" in out.stdout, "the probe reached python but its body was cut short"


def test_the_powershell_probe_runs_the_studio_interpreter():
    # install.ps1 runs setup.ps1 with SKIP_STUDIO_BASE=1 and never puts the venv on PATH,
    # so bare `python` there is the system one: no torchcodec, a silent "absent", and the
    # report never fires on the path every Windows install actually takes. The other
    # bare-`python` probes live inside the base install, which that mode skips.
    text = _SETUP_PS1.read_text(encoding = "utf-8")
    probe = text.index("$_torchcodecProbe = ")
    after = text[probe : text.index('step "torchcodec"', probe)]
    assert '-PythonExe "python"' not in after, "the probe reads whichever python is on PATH"
    assert "$VenvDir" in after


def test_the_shell_probe_carries_no_apostrophe():
    # It is passed as a single-quoted sh string, so one apostrophe anywhere in it
    # (including in a comment) closes the quote and breaks the script.
    text = _SETUP_SH.read_text(encoding = "utf-8")
    start = text.index("_TORCHCODEC_PROBE='") + len("_TORCHCODEC_PROBE='")
    span = text[start : text.index("timeout 60 python -c", start)]
    assert span.count("'") == 1, "the probe body has an apostrophe that closes its own quote"


@pytest.mark.skipif(shutil.which("bash") is None, reason = "bash unavailable")
def test_the_shell_script_is_syntactically_valid():
    # Through stdin rather than a path argument: a Windows bash translates the path
    # in argv and cannot find the file, which fails the test for the wrong reason.
    # Bytes, not text. A text pipe on Windows would re-encode the box characters
    # setup.sh draws its section rules with, and translate every \n back to \r\n,
    # which bash then rejects as a syntax error the real file does not have.
    out = subprocess.run(
        ["bash", "-n"],
        input = _SETUP_SH.read_bytes(),
        capture_output = True,
    )
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
