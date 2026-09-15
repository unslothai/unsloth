# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Setup reports a torchcodec that installed but cannot load.

The wheel is Python-side only: absent FFmpeg avcodec/avutil it still installs and
satisfies notebook_validator's torch/torchcodec matrix, then fails at import, which
`datasets` 4.x reports as "please install 'torchcodec'" for an installed package.
"""

import os
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
    """The probe body as setup.sh ships it. Read, not copied: a copy would drift."""
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
    """Pin the FFmpeg answer: a runner with FFmpeg flips the two loader states."""
    return textwrap.dedent(
        f"""
        import ctypes.util, os
        ctypes.util.find_library = lambda n: {'"/usr/lib/libavutil.so"' if present else 'None'}
        os.environ['PATH'] = ''
        """
    )


def _probe(preamble: str) -> str:
    """The state the installers would read: the sentinel line, not all of stdout."""
    # sys.executable, not a bare "python": a host with only python3 on PATH (Debian
    # without python-is-python3, the AMD CI runners) has no `python` to find.
    # The checkout's own unsloth/import_fixes.py, not whichever unsloth is installed.
    out = subprocess.run(
        [sys.executable, "-c", preamble + _PROBE],
        capture_output = True,
        text = True,
        env = {**os.environ, "PYTHONPATH": str(_STUDIO.parent)},
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
    # Absent implies nothing about FFmpeg, so stay silent. `name` is set the way the
    # import system sets it, since that is what the probe reads.
    assert _probe(_raising("ModuleNotFoundError('no torchcodec', name = 'torchcodec')")) == "absent"


def test_an_unloadable_torchcodec_is_distinguished_from_an_absent_one():
    # ModuleNotFoundError subclasses ImportError: catching ImportError first would
    # collapse the two states this report rests on.
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
    # dep at once. With FFmpeg already on the loader path, advising it is the wrong fix.
    same_error = _raising(
        "RuntimeError('Could not load libtorchcodec. 1. FFmpeg is not properly installed')"
    )
    assert _probe(_ffmpeg(True) + same_error) == "native"
    assert _probe(_ffmpeg(False) + same_error) == "ffmpeg"


def test_a_partial_ffmpeg_is_not_reported_as_ffmpeg_being_present():
    # Distros package the seven linked FFmpeg libraries separately, so a host with only
    # libavutil cannot load torchcodec. Reading that as present sends the user at a torch
    # ABI mismatch when the fix is the rest of FFmpeg.
    partial = textwrap.dedent(
        """
        import ctypes.util, os
        ctypes.util.find_library = (
            lambda n: "/usr/lib/libavutil.so" if n == "avutil" else None
        )
        os.environ['PATH'] = ''
        """
    )
    same_error = _raising(
        "RuntimeError('Could not load libtorchcodec. 1. FFmpeg is not properly installed')"
    )
    assert _probe(partial + same_error) == "ffmpeg"


def test_windows_dll_names_on_a_posix_path_do_not_count(tmp_path):
    # WSL appends the Windows PATH to the Linux one, so a full-shared FFmpeg for Windows
    # is visible there. Its DLLs cannot load into a Linux torchcodec, so they must not
    # turn "install FFmpeg" into "FFmpeg is already on the loader path".
    for lib in ("avutil", "avcodec", "avformat", "avdevice", "avfilter", "swscale", "swresample"):
        (tmp_path / f"{lib}-60.dll").touch()
    dlls = textwrap.dedent(
        f"""
        import ctypes.util, os
        ctypes.util.find_library = lambda n: None
        os.environ['PATH'] = {str(tmp_path)!r}
        os.name = 'posix'
        """
    )
    same_error = _raising(
        "RuntimeError('Could not load libtorchcodec. 1. FFmpeg is not properly installed')"
    )
    assert _probe(dlls + same_error) == "ffmpeg"
    assert _probe(dlls.replace("'posix'", "'nt'") + same_error) == "native"


def test_versioned_libraries_on_the_loader_path_count_as_present(tmp_path):
    # A non-system prefix on LD_LIBRARY_PATH often ships only libavcodec.so.61 and friends:
    # the loader resolves torchcodec's SONAMEs from it, but find_library reads the ld cache
    # and linker names and answers None, which would send the user to install FFmpeg twice.
    for lib in ("avutil", "avcodec", "avformat", "avdevice", "avfilter", "swscale", "swresample"):
        (tmp_path / f"lib{lib}.so.61").touch()
    libs = textwrap.dedent(
        f"""
        import ctypes.util, os
        ctypes.util.find_library = lambda n: None
        os.environ['PATH'] = ''
        os.environ['LD_LIBRARY_PATH'] = {str(tmp_path)!r}
        os.name = 'posix'
        """
    )
    same_error = _raising(
        "RuntimeError('Could not load libtorchcodec. 1. FFmpeg is not properly installed')"
    )
    assert _probe(libs + same_error) == "native"
    assert _probe(libs.replace("'posix'", "'nt'") + same_error) == "ffmpeg"


_IMPORT_FIXES = _STUDIO.parent / "unsloth" / "import_fixes.py"


def test_the_probe_requires_every_ffmpeg_library():
    # All seven, per the shipped libtorchcodec_core*.so NEEDED entries. A subset reports
    # a partial FFmpeg as present and sends the user at the wrong fix.
    text = _IMPORT_FIXES.read_text(encoding = "utf-8")
    body = text[text.index("def _ffmpeg_on_loader_path"): text.index("def torchcodec_load_state")]
    for lib in ("avutil", "avcodec", "avformat", "avdevice", "avfilter", "swscale", "swresample"):
        assert f'"{lib}"' in body, lib


@pytest.mark.parametrize("probe", [_shipped_sh_probe, _shipped_ps1_probe], ids = ["sh", "ps1"])
def test_both_installers_load_the_probe_from_import_fixes(probe):
    # One classification, in unsloth/import_fixes.py, loaded by file: `import unsloth`
    # needs a GPU stack the venv may not have, and a second copy here would drift.
    body = probe()
    assert "import_fixes.py" in body and "spec_from_file_location" in body
    assert "torchcodec_load_state()" in body
    assert "import unsloth\n" not in body and "import torchcodec" not in body


def test_a_missing_transitive_module_is_not_read_as_an_absent_package():
    # Installed, but importing it raises the class an absent one would. Reporting that
    # as absent leaves a damaged install with no warning.
    assert _probe(_raising("ModuleNotFoundError('no numpy', name = 'numpy')")) == "broken"
    # Same for a damaged wheel missing a submodule: the top-level name is the only one
    # an actually-absent package can raise from `import torchcodec`.
    assert _probe(_raising("ModuleNotFoundError('gone', name = 'torchcodec.decoders')")) == "broken"


def test_the_state_is_read_as_a_line_not_as_all_of_stdout():
    # Torch can print to stdout; a whole-buffer read then matches no arm and setup
    # silently drops the report.
    noisy = "import sys; print('banner from a startup hook'); "
    assert (
        _probe(noisy + _raising("ModuleNotFoundError('no torchcodec', name = 'torchcodec')"))
        == "absent"
    )


def test_an_unrelated_import_failure_is_not_blamed_on_ffmpeg():
    # A damaged wheel or a torch ABI mismatch raises here too, and FFmpeg advice would
    # send them at the wrong thing.
    assert _probe(_raising("ImportError('DLL load failed while importing _core')")) == "broken"


@pytest.mark.parametrize("script", [_SETUP_SH, _SETUP_PS1], ids = ["sh", "ps1"])
def test_both_installers_report_the_ffmpeg_case(script):
    # Spelled out past "cannot load", since the non-FFmpeg loader failure opens the same way.
    line = _step_line(script.read_text(encoding = "utf-8"), "installed but cannot load its FFmpeg")
    # Names the real dependency, and what still works, like the whisper.cpp steps do.
    assert "FFmpeg" in line
    assert "soundfile" in line
    # Both fallbacks and the formats they cover, so nobody reads this as "audio is dead".
    assert "PyAV" in line
    assert "wav/flac/mp3/ogg" in line
    assert "m4a/aac/webm" in line


@pytest.mark.parametrize("script", [_SETUP_SH, _SETUP_PS1], ids = ["sh", "ps1"])
def test_both_installers_keep_ffmpeg_advice_out_of_the_other_failure(script):
    line = _step_line(script.read_text(encoding = "utf-8"), "installed but fails to import")
    assert "install an FFmpeg" not in line, "the non-FFmpeg failure still sends them at FFmpeg"
    assert "reinstall torchcodec" in line


@pytest.mark.parametrize("script", [_SETUP_SH, _SETUP_PS1], ids = ["sh", "ps1"])
def test_both_installers_report_the_loader_failure_that_is_not_ffmpeg(script):
    line = _step_line(script.read_text(encoding = "utf-8"), "installed but cannot load its native")
    assert "install an FFmpeg" not in line, "FFmpeg is already there; this sends them at it anyway"
    # EVERY remaining cause: nothing here picks between them, and a subset reads as a
    # diagnosis, sending people to rule out the wrong thing.
    assert "does not support" in line
    assert "torch" in line
    # A missing NPP runtime lands in this same aggregate loader error: see
    # _cuda_major_for_npp in studio/install_python_stack.py.
    assert "NPP" in line
    # Kept in step with unsloth/import_fixes.py (4 through 8): "4 to 7" sent an FFmpeg 8
    # user toward a downgrade that fixes nothing.
    assert "4 to 8" in line, "the supported FFmpeg range must match import_fixes.py"
    assert "4 to 7" not in line


@pytest.mark.parametrize("script", [_SETUP_SH, _SETUP_PS1], ids = ["sh", "ps1"])
def test_the_probe_runs_on_the_fast_update_path_when_a_venv_exists(script):
    # An up-to-date install skips the dependency pass, and that is exactly where a user
    # with an already broken torchcodec lands on every update. Probe whenever there is a
    # venv to inspect; skip only when nothing was installed and none exists (Colab, first run).
    # The ENCLOSING condition, not "the name appears somewhere above": the assignment sits
    # hundreds of lines earlier, so a backwards search passes even with the guard deleted.
    text = script.read_text(encoding = "utf-8")
    opener, skip, venv = {
        ".sh": ("_TORCHCODEC_PROBE=", "_SKIP_PYTHON_DEPS", '-x "$VENV_DIR/bin/python"'),
        ".ps1": ("$_torchcodecProbe = @", "SkipPythonDeps", "Join-Path $VenvDir 'Scripts\\python.exe'"),
    }[script.suffix]
    probe = text.index(opener)
    guard = text.rfind("if ", 0, probe)
    line = text[guard : text.index("\n", guard)]
    assert skip in line, f"the probe's enclosing condition does not test the skip-python-deps flag: {line!r}"
    assert venv in line, f"the probe's enclosing condition does not fall back to an existing venv: {line!r}"


def test_the_shell_probe_is_skipped_in_llama_only_mode():
    # _SKIP_PYTHON_DEPS is assigned in the base install, which llama-only skips, so under
    # `set -u` a bare read aborts the run. The gate keeps the slow probe off the update
    # path; the default expansion keeps the unbound read from coming back.
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
    assert 'timeout 60 "$_TORCHCODEC_PY" -c' in after
    # ...and still runs where coreutils `timeout` is absent, as the GPU probes do.
    assert "command -v timeout" in after
    # Stock macOS takes that fallback, so the deadline has to live in the body too.
    assert "_alarm(60)" in _shipped_sh_probe()


@pytest.mark.parametrize("script", [_SETUP_SH, _SETUP_PS1], ids = ["sh", "ps1"])
def test_both_installers_read_the_state_as_a_line(script):
    # A torch banner ahead of the answer leaves a whole-buffer read matching no arm,
    # and the report vanishes.
    text = script.read_text(encoding = "utf-8")
    probe = text.index("TORCHCODEC=")
    after = text[probe : text.index('step "torchcodec"', probe)]
    assert "s/^TORCHCODEC=//p" in after or "(?m)^TORCHCODEC=" in after


def test_the_two_installers_ship_the_same_probe():
    # Separate copies drift and the two platforms then report differently for the same
    # install. Quote style is the one allowed difference: setup.sh wraps the body in
    # single quotes, so its Python strings must use double ones.
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
    # Invoke-BoundedPythonProbe wraps the body in double quotes for -c <body>, so one
    # more anywhere (comments included) closes it early and python runs a truncated head
    # that still parses: exit 0, no output, and the report goes quiet.
    assert '"' not in _shipped_ps1_probe()


# Runs the shipped helper against the shipped body, so neither can drift from the test.
# $args: setup.ps1, the interpreter, a PYTHONPATH to import from.
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
    # The unloadable case on purpose: a truncated body keeps the ModuleNotFoundError
    # branch, so an absent torchcodec answers the same either way and proves nothing.
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
            os.pathsep.join([str(tmp_path), str(_STUDIO.parent)]),
        ],
        capture_output = True,
        text = True,
    )
    assert out.returncode == 0, out.stderr
    # Either answer proves the point: both come from the `except Exception` arm a
    # truncated body would lose. Which one depends on the machine's own FFmpeg, so
    # pinning to "ffmpeg" fails on any box that has it.
    assert any(
        f"TORCHCODEC={state}" in out.stdout for state in ("ffmpeg", "native")
    ), f"the probe reached python but its body was cut short: {out.stdout!r}"


def test_the_powershell_probe_runs_the_studio_interpreter():
    # install.ps1 runs setup.ps1 with SKIP_STUDIO_BASE=1 and never puts the venv on PATH,
    # so bare `python` is the system one: a silent "absent" on the path every Windows
    # install actually takes.
    text = _SETUP_PS1.read_text(encoding = "utf-8")
    probe = text.index("$_torchcodecProbe = ")
    after = text[probe : text.index('step "torchcodec"', probe)]
    assert '-PythonExe "python"' not in after, "the probe reads whichever python is on PATH"
    assert "$VenvDir" in after


def test_the_shell_probe_runs_the_studio_interpreter():
    # The uv installer branch prepends $HOME/.local/bin after the venv activation, so a
    # pyenv/pipx/asdf shim shadows it and bare `python` answers a silent "absent". Fall
    # back to bare `python` only where no venv exists (Colab).
    text = _SETUP_SH.read_text(encoding = "utf-8")
    probe = text.index("_TORCHCODEC_PROBE=")
    after = text[probe : text.index('step "torchcodec"', probe)]
    assert " python -c " not in after, "the probe reads whichever python is on PATH"
    assert '_TORCHCODEC_PY="$VENV_DIR/bin/python"' in after
    assert '"$_TORCHCODEC_PY" -c "$_TORCHCODEC_PROBE"' in after
    assert '_TORCHCODEC_PY="python"' in after, "no venv (Colab) must still have an interpreter"


def test_the_shell_probe_carries_no_apostrophe():
    # Passed as a single-quoted sh string: one apostrophe anywhere, comments included,
    # closes the quote and breaks the script.
    text = _SETUP_SH.read_text(encoding = "utf-8")
    start = text.index("_TORCHCODEC_PROBE='") + len("_TORCHCODEC_PROBE='")
    span = text[start : text.index("_TORCHCODEC_PY=", start)]
    assert span.count("'") == 1, "the probe body has an apostrophe that closes its own quote"


def _bash_runs() -> bool:
    # On Windows `bash` on PATH may be the WSL stub, which prints a banner and exits 1.
    if shutil.which("bash") is None:
        return False
    try:
        out = subprocess.run(["bash", "-c", "echo ok"], capture_output = True, timeout = 30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and out.stdout.strip() == b"ok"


@pytest.mark.skipif(not _bash_runs(), reason = "no working bash")
def test_the_shell_script_is_syntactically_valid():
    # Through stdin, since a Windows bash translates a path argument and cannot find the
    # file. Bytes, not text: a text pipe on Windows re-encodes setup.sh box characters and
    # turns every \n back into \r\n, which bash rejects as a syntax error.
    out = subprocess.run(
        ["bash", "-n"],
        input = _SETUP_SH.read_bytes(),
        capture_output = True,
    )
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
