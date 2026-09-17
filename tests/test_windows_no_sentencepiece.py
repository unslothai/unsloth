# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""On Windows, sentencepiece is never imported.

The point is not that transformers reports it unavailable, it is that the compiled extension
is never handed to the Windows loader. A code integrity policy refuses by reputation, one file
at a time, and the refusal is a Bad Image dialog: any probe that asks whether this machine
would refuse the file has already produced the thing being avoided.
"""

import importlib.util
import os
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from unsloth.import_fixes import (  # noqa: E402
    DISABLE_SENTENCEPIECE_VARIABLE,
    disable_sentencepiece_on_windows,
    sentencepiece_should_be_disabled,
)

BACKEND = REPO / "studio" / "backend"
MAIN = BACKEND / "main.py"
GUARD = BACKEND / "utils" / "sentencepiece_guard.py"


def _studio_guard():
    """Studio's copy, loaded by path so the test needs nothing else from the backend tree."""
    spec = importlib.util.spec_from_file_location("studio_sentencepiece_guard", GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(DISABLE_SENTENCEPIECE_VARIABLE, raising = False)
    # The rule declines once transformers is imported, and the test session has imported it.
    # Removed here so each test states its own starting point; the one test that wants it
    # present puts it back.
    monkeypatch.delitem(sys.modules, "transformers", raising = False)
    yield


@pytest.mark.parametrize(
    "platform,expected", [("win32", True), ("linux", False), ("darwin", False)]
)
def test_the_default_is_windows_only(platform, expected, monkeypatch):
    """WSL reports linux and is deliberately in the second group: App Control does not enforce
    over ELF binaries in the guest, so there is no extension for it to refuse there."""
    monkeypatch.setattr(sys, "platform", platform)
    assert sentencepiece_should_be_disabled() is expected


@pytest.mark.parametrize("value", ["1", "true", "YES", " On ", "TRUE"])
def test_a_truthy_flag_disables_on_any_platform(value, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv(DISABLE_SENTENCEPIECE_VARIABLE, value)
    assert sentencepiece_should_be_disabled() is True


@pytest.mark.parametrize("value", ["0", "false", "NO", " off "])
def test_a_falsy_flag_opts_windows_back_in(value, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv(DISABLE_SENTENCEPIECE_VARIABLE, value)
    assert sentencepiece_should_be_disabled() is False


@pytest.mark.parametrize("value", ["maybe", "2", "", "   ", "disable"])
def test_an_unrecognised_value_is_the_platform_default(value, monkeypatch):
    """This runs at the top of the process. A typo in an environment variable must not be
    fatal, and must not silently mean the opposite of what was typed."""
    monkeypatch.setenv(DISABLE_SENTENCEPIECE_VARIABLE, value)
    monkeypatch.setattr(sys, "platform", "win32")
    assert sentencepiece_should_be_disabled() is True
    monkeypatch.setattr(sys, "platform", "linux")
    assert sentencepiece_should_be_disabled() is False


def test_it_installs_the_sentinel_and_the_import_then_fails_like_an_absent_package(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "sentencepiece", raising = False)

    assert disable_sentencepiece_on_windows() is True
    assert sys.modules["sentencepiece"] is None

    with pytest.raises(ImportError):
        # ModuleNotFoundError, which is an ImportError, so every `try/except ImportError`
        # already in transformers handles it as "not installed".
        import sentencepiece  # noqa: F401


def test_nothing_happens_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delitem(sys.modules, "sentencepiece", raising = False)
    assert disable_sentencepiece_on_windows() is False
    assert "sentencepiece" not in sys.modules


def test_an_already_imported_sentencepiece_is_left_alone(monkeypatch):
    """Replacing a live module would break whoever is holding a reference to it, and by this
    point the extension has already loaded, so there is nothing left to prevent."""
    sentinel = object()
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "sentencepiece", sentinel)
    assert disable_sentencepiece_on_windows() is False
    assert sys.modules["sentencepiece"] is sentinel


def test_it_declines_once_transformers_is_imported(monkeypatch):
    """Installing it late is worse than not installing it at all.

    transformers reads availability from find_spec during its own import and caches it, so a
    sentinel added afterwards only makes the two disagree: it reports the package available
    and the import then fails. Measured on 4.57.6, unsloth/gemma-2-2b-it loads with the rule
    applied in time and without the rule at all, and raises ModuleNotFoundError with the rule
    applied afterwards. Whoever imported transformers first keeps the ordinary behaviour.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "sentencepiece", raising = False)
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))

    assert disable_sentencepiece_on_windows() is False
    assert "sentencepiece" not in sys.modules

    studio = _studio_guard()
    monkeypatch.setattr(
        studio,
        "sys",
        types.SimpleNamespace(platform = "win32", modules = {"transformers": object()}),
    )
    assert studio.disable_sentencepiece_on_windows() is False
    assert "sentencepiece" not in studio.sys.modules


def test_a_sentinel_installed_in_time_survives_a_later_transformers_import(monkeypatch):
    """The late check must not undo the ordinary case, where the rule ran first and
    transformers was imported after it."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "sentencepiece", None)
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    assert disable_sentencepiece_on_windows() is True
    assert sys.modules["sentencepiece"] is None


def test_calling_it_twice_is_stable(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "sentencepiece", raising = False)
    assert disable_sentencepiece_on_windows() is True
    assert disable_sentencepiece_on_windows() is True
    assert sys.modules["sentencepiece"] is None


def test_the_studio_parent_applies_the_same_rule_without_importing_unsloth():
    """Studio's parent must not import unsloth: that runs unsloth/__init__.py, whose GPU branch
    pulls torch, Triton, transformers and the model stack into a long-lived process built to
    stay light, and can open a competing GPU context. So it calls Studio's own copy instead."""
    source = MAIN.read_text(encoding = "utf-8")
    assert "from unsloth.import_fixes import" not in source
    assert "import unsloth\n" not in source

    marker = source.index('os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")')
    guard = source.index("from utils.sentencepiece_guard import")
    assert guard > marker

    # Real import statements only. Matching the bare word finds the surrounding comments,
    # which say nothing about execution order.
    imports = [
        m.start()
        for m in re.finditer(r"^\s*(?:import transformers|from transformers)", source, re.M)
    ]
    assert not imports or guard < min(imports), (
        "the sentinel must be installed before anything imports transformers, which reads "
        "sentencepiece availability during its own import"
    )


@pytest.mark.parametrize(
    "platform,env,expect_disabled",
    [
        ("win32", None, True),
        ("win32", "0", False),
        ("linux", None, False),
        ("linux", "1", True),
    ],
)
def test_the_two_spellings_agree(platform, env, expect_disabled, monkeypatch):
    """The package helper and Studio's copy, driven through the same cases. Compared by
    behaviour rather than by source text, which would pass on two implementations that had
    quietly stopped agreeing."""
    monkeypatch.setattr(sys, "platform", platform)
    if env is None:
        monkeypatch.delenv(DISABLE_SENTENCEPIECE_VARIABLE, raising = False)
    else:
        monkeypatch.setenv(DISABLE_SENTENCEPIECE_VARIABLE, env)
    assert sentencepiece_should_be_disabled() is expect_disabled

    studio = _studio_guard()
    monkeypatch.setattr(studio.sys, "platform", platform)
    assert studio.DISABLE_SENTENCEPIECE_VARIABLE == DISABLE_SENTENCEPIECE_VARIABLE
    assert studio.sentencepiece_should_be_disabled() is expect_disabled

    # A stub sys, because the real one already has sentencepiece imported by the test session,
    # and the rule correctly declines to replace a live module. Running it against the real
    # sys.modules would test the fixture, not the rule.
    monkeypatch.setattr(studio, "sys", types.SimpleNamespace(platform = platform, modules = {}))
    assert studio.disable_sentencepiece_on_windows() is expect_disabled
    assert (studio.sys.modules.get("sentencepiece", "absent") is None) is expect_disabled


def _run_shared_entrypoint(tmp_path, env):
    """Drive the workers' shared spawn entrypoint against a stand-in worker module.

    The stand-in records, at its own module scope, what the interpreter looked like when the
    entrypoint imported it. That is the moment under test: the real worker modules import
    transformers from there onwards.
    """
    (tmp_path / "sentencepiece_entrypoint_probe.py").write_text(
        textwrap.dedent(
            """
            import os, sys
            AT_IMPORT = sys.modules.get("sentencepiece", "absent") is None
            ENV = os.environ.get("UNSLOTH_STUDIO_SP_PROBE")

            def report():
                print("SENTINEL AT IMPORT", AT_IMPORT)
                print("ENV APPLIED", ENV)
            """
        ),
        encoding = "utf-8",
    )
    program = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(BACKEND)!r})
        sys.path.insert(0, {str(tmp_path)!r})
        from utils.native_path_leases import run_without_native_path_secret
        assert "sentencepiece" not in sys.modules, sys.modules["sentencepiece"]
        run_without_native_path_secret(
            "sentencepiece_entrypoint_probe", "report", {{"UNSLOTH_STUDIO_SP_PROBE": "yes"}}
        )
        """
    )
    return subprocess.run(
        [sys.executable, "-c", program],
        capture_output = True,
        text = True,
        timeout = 300,
        env = env,
    )


def test_the_shared_worker_entrypoint_installs_it_before_the_worker_module(tmp_path):
    """Every Studio worker is a spawned interpreter that inherits no sys.modules, and each one
    imports transformers (version activation, fast-path hooks) long before it imports unsloth.
    A sentinel installed after that leaves transformers reporting sentencepiece available while
    importing it fails, which breaks tokenizer loads that work either without the rule or with
    it applied in time. So the shared entrypoint installs it before the worker module."""
    out = _run_shared_entrypoint(tmp_path, {**os.environ, DISABLE_SENTENCEPIECE_VARIABLE: "1"})
    assert "SENTINEL AT IMPORT True" in out.stdout, (out.stdout, out.stderr[-2000:])
    # The captured cache environment still lands first: the rule reads the environment.
    assert "ENV APPLIED yes" in out.stdout, (out.stdout, out.stderr[-2000:])


def test_the_shared_worker_entrypoint_leaves_it_alone_when_not_asked(tmp_path):
    """The same entrypoint where the rule does not apply: off Windows and without the flag, a
    worker still gets the real package."""
    env = {k: v for k, v in os.environ.items() if k != DISABLE_SENTENCEPIECE_VARIABLE}
    out = _run_shared_entrypoint(tmp_path, env)
    expected = "SENTINEL AT IMPORT " + str(sys.platform == "win32")
    assert expected in out.stdout, (out.stdout, out.stderr[-2000:])


@pytest.mark.skipif(
    __import__("importlib.util", fromlist = ["util"]).find_spec("transformers") is None,
    reason = "transformers is not installed",
)
def test_transformers_reports_it_absent_and_never_loads_the_extension():
    """The whole point, end to end, in a clean interpreter.

    transformers derives availability from find_spec, which finds the sentinel and answers
    False on its own, so nothing here monkey patches is_sentencepiece_available. That matters:
    a flag that lies while the package is still importable is a different and worse state, and
    on 4.52 through 4.57 it sends tokenizer_class_from_name into a fallback that imports the
    slow tokenizer module and reaches its unguarded `import sentencepiece as spm`.
    """
    program = textwrap.dedent(
        """
        import sys, warnings
        warnings.filterwarnings("ignore")
        sys.modules["sentencepiece"] = None
        import transformers
        from transformers.utils import import_utils
        loaded = [m for m in sys.modules
                  if m == "sentencepiece" or m.startswith("sentencepiece.")]
        alive = [m for m in loaded if sys.modules[m] is not None]
        print("AVAILABLE", import_utils.is_sentencepiece_available())
        print("ALIVE", alive)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", program],
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert "AVAILABLE False" in out.stdout, (out.stdout, out.stderr[-2000:])
    assert "ALIVE []" in out.stdout, (
        "the compiled extension must never be loaded",
        out.stdout,
        out.stderr[-2000:],
    )
