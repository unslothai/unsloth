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

MAIN = REPO / "studio" / "backend" / "main.py"


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(DISABLE_SENTENCEPIECE_VARIABLE, raising = False)
    yield


@pytest.mark.parametrize("platform,expected", [("win32", True), ("linux", False), ("darwin", False)])
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


def test_calling_it_twice_is_stable(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "sentencepiece", raising = False)
    assert disable_sentencepiece_on_windows() is True
    assert disable_sentencepiece_on_windows() is True
    assert sys.modules["sentencepiece"] is None


def test_the_studio_parent_applies_the_same_rule_without_importing_unsloth():
    """Studio's parent must not import unsloth: that runs unsloth/__init__.py, whose GPU branch
    pulls torch, Triton, transformers and the model stack into a long-lived process built to
    stay light, and can open a competing GPU context. So the rule is inlined there, and this
    holds the two spellings to the same behaviour."""
    source = MAIN.read_text(encoding = "utf-8")
    assert 'sys.modules["sentencepiece"] = None' in source
    assert "from unsloth.import_fixes import" not in source
    assert "import unsloth\n" not in source

    marker = source.index('os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")')
    guard = source.index('sys.modules["sentencepiece"] = None')
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


@pytest.mark.parametrize("platform,env,expect_disabled", [
    ("win32", None, True),
    ("win32", "0", False),
    ("linux", None, False),
    ("linux", "1", True),
])
def test_the_two_spellings_agree(platform, env, expect_disabled, monkeypatch):
    """The package helper and the inlined Studio condition, driven through the same cases.
    Compared by behaviour rather than by source text, which would pass on two implementations
    that had quietly stopped agreeing."""
    monkeypatch.setattr(sys, "platform", platform)
    if env is None:
        monkeypatch.delenv(DISABLE_SENTENCEPIECE_VARIABLE, raising = False)
    else:
        monkeypatch.setenv(DISABLE_SENTENCEPIECE_VARIABLE, env)
    assert sentencepiece_should_be_disabled() is expect_disabled

    source = MAIN.read_text(encoding = "utf-8")
    start = source.index("_DISABLE_SENTENCEPIECE = ")
    end = source.index('sys.modules["sentencepiece"] = None', start) + len(
        'sys.modules["sentencepiece"] = None'
    )
    inlined = textwrap.dedent(source[start:end])
    # A stub sys, because the real one already has sentencepiece imported by the test session,
    # and the snippet correctly declines to replace a live module. Running it against the real
    # sys.modules would test the fixture, not the rule.
    stub = types.SimpleNamespace(platform = platform, modules = {})
    scope = {"os": os, "sys": stub}
    exec(compile(inlined, "<studio-main-inline>", "exec"), scope)
    assert (stub.modules.get("sentencepiece", "absent") is None) is expect_disabled


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
        [sys.executable, "-c", program], capture_output = True, text = True, timeout = 600,
    )
    assert "AVAILABLE False" in out.stdout, (out.stdout, out.stderr[-2000:])
    assert "ALIVE []" in out.stdout, (
        "the compiled extension must never be loaded", out.stdout, out.stderr[-2000:],
    )
