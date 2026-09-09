# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Smart App Control blocking sentencepiece's compiled extension.

Reported from a Windows machine as "Part of this app has been blocked / Some features of
Python may not work because we can't confirm who published
_sentencepiece.cp313-win_amd64.pyd that the app tried to load."

``transformers._is_package_available`` decides availability from ``find_spec`` plus
installed metadata, and neither loads the extension, so ``is_sentencepiece_available()``
answers True on that machine while every import of it raises. The loudest consequence is
a tokenizer that will not build; the quiet one is Studio's ``get_native_chat_template``,
which catches any exception, logs a warning and returns None, leaving the model to
generate under a substituted chat template.

What is asserted here is the shape of the correction rather than the block itself. As a
temporary measure the guard now fires on Windows by default, so the extension is never
loaded there at all; ``UNSLOTH_DISABLE_SENTENCEPIECE=0`` opts back in, and a machine that
has opted back in is still protected when the import really fails. Off Windows nothing
happens unless the flag is set truthy.
"""

import importlib
import sys
import warnings
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from unsloth import import_fixes  # noqa: E402


VARIABLE = import_fixes._DISABLE_SENTENCEPIECE_VARIABLE


@pytest.fixture(autouse = True)
def _reset_guard(monkeypatch):
    """The guard caches its verdict for the process; each test starts from unknown.

    The flag is cleared too, so an operator who exported it in the shell running pytest
    cannot change what these tests mean.
    """
    import_fixes._SENTENCEPIECE_GUARD_RESULT = None
    monkeypatch.delenv(VARIABLE, raising = False)
    yield
    import_fixes._SENTENCEPIECE_GUARD_RESULT = None


@pytest.fixture
def transformers_flag():
    """transformers' availability flag, restored afterwards."""
    module = importlib.import_module("transformers.utils.import_utils")
    original_flag = module._sentencepiece_available
    original_function = module.is_sentencepiece_available
    yield module
    module._sentencepiece_available = original_flag
    module.is_sentencepiece_available = original_function


def _blocked_import(monkeypatch, *, winerror = 577):
    """A Windows box whose sentencepiece is installed and refuses to load.

    With the flag set falsy, because that is the only configuration in which the import
    probe still runs: unset, Windows disables sentencepiece by policy and never touches
    the extension, so a blocked machine is already covered before this path is reached.
    """
    monkeypatch.setenv(VARIABLE, "0")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "sentencepiece", raising = False)

    class _Spec:
        origin = r"C:\\Users\\u\\.unsloth\\studio\\Lib\\site-packages\\sentencepiece\\__init__.py"

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: _Spec())

    def _raise(name, *args, **kwargs):
        error = ImportError(
            "DLL load failed while importing _sentencepiece: "
            "Windows cannot verify the digital signature for this file."
        )
        error.winerror = winerror
        raise error

    monkeypatch.setattr(importlib, "import_module", _raise)


def test_a_blocked_extension_makes_transformers_say_sentencepiece_is_absent(
    monkeypatch, transformers_flag
):
    """The correction itself. transformers then takes the path it takes on a machine
    where sentencepiece was never installed, which is supported for every model that
    ships tokenizer.json, and the models that truly need it fail with the backend
    message that names it rather than a loader error from inside the tokenizer."""
    _blocked_import(monkeypatch)
    transformers_flag._sentencepiece_available = True

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert import_fixes.disable_sentencepiece_if_blocked() is True

    assert transformers_flag.is_sentencepiece_available() is False
    message = str(caught[0].message)
    assert "sentencepiece" in message
    assert "577" in message, "the loader error is what tells the user it was a block"


def test_a_working_sentencepiece_is_left_alone(monkeypatch, transformers_flag):
    """A Windows machine that has opted back in: an import that succeeds must leave the
    flag exactly where it was, or opting in would achieve nothing. The opt-in is required
    here, since with the flag unset Windows disables sentencepiece by policy whether or
    not the extension would have loaded."""
    monkeypatch.setenv(VARIABLE, "0")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "sentencepiece", raising = False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(importlib, "import_module", lambda name, *a, **k: object())
    transformers_flag._sentencepiece_available = True

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert import_fixes.disable_sentencepiece_if_blocked() is False

    assert transformers_flag.is_sentencepiece_available() is True
    assert caught == []


def test_windows_disables_by_default_without_touching_the_extension(
    monkeypatch, transformers_flag
):
    """The temporary Windows default, against the transformers actually installed here.

    Disabled because the platform is Windows, not because anything was probed: the point
    is that the extension is never loaded, so a machine whose loader would refuse it never
    gives the loader the chance. Any import at all is the failure.

    UNSLOTH_DISABLE_SENTENCEPIECE=0 is the way back, and is covered above.
    """
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "sentencepiece", raising = False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    def _explode(name, *args, **kwargs):
        raise AssertionError("the Windows default must not import sentencepiece")

    monkeypatch.setattr(importlib, "import_module", _explode)
    transformers_flag._sentencepiece_available = True

    assert import_fixes.sentencepiece_disabled_by_policy() == "windows"
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert import_fixes.disable_sentencepiece_if_blocked() is True

    assert transformers_flag.is_sentencepiece_available() is False
    assert caught == [], (
        "every healthy Windows launch would print this, so it is logged rather than "
        "warned; only a real block earns a warning"
    )


def test_a_sentencepiece_that_is_not_installed_is_not_a_block(monkeypatch, transformers_flag):
    """find_spec answering None is the ordinary uninstalled case. transformers already
    reports it correctly, so there is nothing to correct and nothing to warn about. Asked
    with the flag off, which is where "is this a block?" is still a question."""
    monkeypatch.setenv(VARIABLE, "0")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "sentencepiece", raising = False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert import_fixes.sentencepiece_import_error() is None
    assert import_fixes.disable_sentencepiece_if_blocked() is False


def test_the_guard_never_costs_an_import_off_windows(monkeypatch):
    """Nobody has reported this off Windows, and the probe is an import: paying it on
    Linux and macOS would slow every session to answer a question already answered."""
    monkeypatch.setattr(sys, "platform", "linux")

    def _explode(name, *args, **kwargs):
        raise AssertionError("the guard must not import anything off Windows")

    monkeypatch.setattr(importlib, "import_module", _explode)
    assert import_fixes.sentencepiece_import_error() is None
    assert import_fixes.disable_sentencepiece_if_blocked() is False


def test_an_already_imported_sentencepiece_is_not_reimported(monkeypatch):
    """It is in sys.modules, so it loaded; probing again would only be a way to get a
    different answer than the process is already running under."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "sentencepiece", object())

    def _explode(name, *args, **kwargs):
        raise AssertionError("already imported, so there is nothing to probe")

    monkeypatch.setattr(importlib, "import_module", _explode)
    assert import_fixes.sentencepiece_import_error() is None


def test_the_smart_app_control_read_answers_rather_than_raising(monkeypatch):
    """It reads HKLM\\SYSTEM\\CurrentControlSet\\Control\\CI\\Policy, which a standard
    user can read, so it needs no elevation and prompts for nothing. Every "cannot say"
    is None, including this machine, which is not Windows."""
    assert import_fixes.smart_app_control_state() is None

    monkeypatch.setattr(sys, "platform", "win32")
    fake = type(sys)("winreg")
    fake.HKEY_LOCAL_MACHINE = 0
    fake.KEY_READ = 0

    def _open_key(*args, **kwargs):
        raise PermissionError("denied")

    fake.OpenKey = _open_key
    fake.QueryValueEx = lambda *a, **k: (0, 0)
    monkeypatch.setitem(sys.modules, "winreg", fake)
    assert import_fixes.smart_app_control_state() is None, "a denied read is unknown, not off"


def test_the_state_is_not_what_decides(monkeypatch, transformers_flag):
    """Smart App Control blocks by reputation, one file at a time, so it is on for many
    machines whose sentencepiece loads perfectly and off on machines where an antivirus
    refuses the same file. Deciding on the state rather than on the import would take
    the tokenizer away from the first group and leave the second broken. Asked with the
    flag off, since that is the only configuration in which the probe still runs."""
    monkeypatch.setenv(VARIABLE, "0")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "sentencepiece", raising = False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(importlib, "import_module", lambda name, *a, **k: object())
    monkeypatch.setattr(import_fixes, "smart_app_control_state", lambda: 1)
    transformers_flag._sentencepiece_available = True

    assert import_fixes.disable_sentencepiece_if_blocked() is False
    assert transformers_flag.is_sentencepiece_available() is True


def test_the_guard_runs_at_import(monkeypatch):
    """A fix nobody calls is not a fix. Read off the source rather than by importing
    unsloth a second time, which costs seconds and cannot be undone in-process."""
    source = (PACKAGE_ROOT / "unsloth" / "__init__.py").read_text(encoding = "utf-8")
    assert "disable_sentencepiece_if_blocked" in source
    assert source.count("_guard_sentencepiece()") == 1


def test_the_guard_runs_before_anything_resolves_autotokenizer():
    """Codex 3964051672, P2. transformers freezes derived state at import, so ordering is
    the whole fix rather than a detail of it.

    models/auto/tokenization_auto.py evaluates ``is_sentencepiece_available()`` at module
    scope in 5.x, twice over: once to decide whether to import SentencePieceBackend, and
    once per sentencepiece-only entry of TOKENIZER_MAPPING_NAMES, as in
    ``("marian", "MarianTokenizer" if is_sentencepiece_available() else None)``. Those
    values are materialised at that moment and no later rebinding of the function reaches
    them, so a guard running after AutoTokenizer resolves leaves a Marian or M2M100 load
    importing the blocked extension and raising the loader error the guard exists to
    replace.

    _gpu_init resolves AutoTokenizer on the GPU path, so the guard has to precede it.
    """
    source = (PACKAGE_ROOT / "unsloth" / "__init__.py").read_text(encoding = "utf-8")
    guard_at = source.index("_guard_sentencepiece()")
    gpu_init_at = source.index("from ._gpu_init import")
    assert guard_at < gpu_init_at, (
        "the sentencepiece guard must run before _gpu_init resolves AutoTokenizer, or "
        "TOKENIZER_MAPPING_NAMES is already frozen as available"
    )
