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
import textwrap
from pathlib import Path

import pytest

_STUDIO = Path(__file__).resolve().parents[2]
_SETUP_SH = _STUDIO / "setup.sh"
_SETUP_PS1 = _STUDIO / "setup.ps1"

_PROBE = textwrap.dedent(
    """
    try:
        import torchcodec  # noqa: F401
    except ModuleNotFoundError:
        print("absent")
    except Exception:
        print("unloadable")
    else:
        print("ok")
    """
)


def _probe(preamble: str) -> str:
    """Run the probe's logic with torchcodec's import forced into one outcome."""
    out = subprocess.run(
        ["python", "-c", preamble + _PROBE],
        capture_output = True,
        text = True,
    )
    return out.stdout.strip()


def test_a_missing_torchcodec_is_not_reported_as_an_ffmpeg_problem():
    # Nothing to say: absent says nothing about FFmpeg, and the soundfile path stands.
    preamble = "import sys; sys.modules['torchcodec'] = None; del sys.modules['torchcodec']\n"
    assert _probe(preamble + "import builtins\n") in {"absent", "ok", "unloadable"}


def test_an_unloadable_torchcodec_is_distinguished_from_an_absent_one():
    # The distinction the report rests on. ModuleNotFoundError is a subclass of
    # ImportError, so catching ImportError first would collapse the two.
    loadable = _probe(
        "import sys, types; sys.modules['torchcodec'] = types.ModuleType('torchcodec')\n"
    )
    assert loadable == "ok"

    unloadable = _probe(
        "import sys, builtins\n"
        "_real = builtins.__import__\n"
        "def _fake(name, *a, **k):\n"
        "    if name == 'torchcodec':\n"
        "        raise RuntimeError('Could not load libtorchcodec')\n"
        "    return _real(name, *a, **k)\n"
        "builtins.__import__ = _fake\n"
    )
    assert unloadable == "unloadable"

    absent = _probe(
        "import sys, builtins\n"
        "_real = builtins.__import__\n"
        "def _fake(name, *a, **k):\n"
        "    if name == 'torchcodec':\n"
        "        raise ModuleNotFoundError('no torchcodec')\n"
        "    return _real(name, *a, **k)\n"
        "builtins.__import__ = _fake\n"
    )
    assert absent == "absent"


@pytest.mark.parametrize("script", [_SETUP_SH, _SETUP_PS1], ids = ["sh", "ps1"])
def test_both_installers_report_the_unloadable_case(script):
    text = script.read_text(encoding = "utf-8")
    assert "unloadable" in text, "the installer does not distinguish the case"
    warn = text.index('step "torchcodec" "installed but cannot load')
    # Names the real dependency, and what still works, like the whisper.cpp steps do.
    line = text[warn : text.index("\n", warn)]
    assert "FFmpeg" in line
    assert "soundfile" in line
    # The formats libsndfile actually covers, so nobody reads this as "audio is dead".
    assert "wav/flac/mp3/ogg" in line


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


@pytest.mark.skipif(shutil.which("bash") is None, reason = "bash unavailable")
def test_the_shell_heredoc_is_syntactically_valid():
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
