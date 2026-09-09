# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The two copies of the blocked-sentencepiece guard, held to the same behaviour.

There are two on purpose. ``unsloth.import_fixes`` carries it for people who installed
the pip package, where importing unsloth is the point. ``studio/backend/utils`` carries
it because the Studio parent must not import unsloth: that runs ``unsloth/__init__.py``,
whose GPU branch pulls torch, Triton, transformers and the model stack into a
long-lived process that exists to stay light, and can open a competing GPU context.

Two copies drift. These tests are the thing that stops them, and they drive both through
the same cases rather than comparing source text, which would pass on two
implementations that had quietly stopped agreeing about what they do.
"""

import importlib
import sys
import types
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for candidate in (REPO, REPO / "studio" / "backend"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from unsloth import import_fixes as PACKAGE_COPY  # noqa: E402
from utils import sentencepiece_guard as STUDIO_COPY  # noqa: E402

# (module, probe, patcher, policy reader, entry point), so each case runs on both.
COPIES = [
    pytest.param(
        PACKAGE_COPY,
        "sentencepiece_import_error",
        "_tell_transformers_sentencepiece_is_absent",
        "smart_app_control_state",
        "disable_sentencepiece_if_blocked",
        "_SENTENCEPIECE_GUARD_RESULT",
        id = "unsloth.import_fixes",
    ),
    pytest.param(
        STUDIO_COPY,
        "sentencepiece_import_error",
        "tell_transformers_sentencepiece_is_absent",
        "smart_app_control_state",
        "disable_sentencepiece_if_blocked",
        "_RESULT",
        id = "studio.utils.sentencepiece_guard",
    ),
]


@pytest.fixture(autouse = True)
def _reset():
    PACKAGE_COPY._SENTENCEPIECE_GUARD_RESULT = None
    STUDIO_COPY._RESULT = None
    yield
    PACKAGE_COPY._SENTENCEPIECE_GUARD_RESULT = None
    STUDIO_COPY._RESULT = None


def _blocked_import(monkeypatch, winerror = 577):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "sentencepiece", raising = False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    def _raise(name, *args, **kwargs):
        error = ImportError(
            "DLL load failed while importing _sentencepiece: "
            "Windows cannot verify the digital signature for this file."
        )
        error.winerror = winerror
        raise error

    monkeypatch.setattr(importlib, "import_module", _raise)


def _transformers_5x():
    """The 5.5.0 shape: an lru_cache function, no global, and captured consumers."""
    import functools

    module = types.ModuleType("fake_import_utils")

    @functools.lru_cache
    def is_sentencepiece_available():
        return True

    module.is_sentencepiece_available = is_sentencepiece_available
    module.BACKENDS_MAPPING = {
        "sentencepiece": (is_sentencepiece_available, "pip install sentencepiece"),
    }
    return module


def _transformers_4x():
    """The 4.57 shape: a module global the function returns."""
    module = types.ModuleType("fake_import_utils_4")
    module._sentencepiece_available = True
    module.is_sentencepiece_available = lambda: module._sentencepiece_available
    return module


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
def test_the_error_codes_are_the_same(module, probe, patcher, policy, entry, cache):
    """Drift here would mean one copy calls a block what the other calls a plain failure."""
    codes = getattr(module, "BLOCKED_IMAGE_WINERRORS", None) or getattr(
        module, "_BLOCKED_IMAGE_WINERRORS"
    )
    assert set(codes) == {225, 577, 1260}


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
def test_a_transformers_5x_layout_is_corrected_through_its_captured_consumers(
    module, probe, patcher, policy, entry, cache
):
    """The shape Studio actually installs. is_sentencepiece_available is lru_cache-wrapped
    around _is_package_available, which asks find_spec and loads nothing, so it answers
    True while the extension is refused; and BACKENDS_MAPPING plus every module that did
    a `from ... import` hold the original object, so replacing only the defining module
    leaves them all still saying True."""
    import_utils = _transformers_5x()
    original = import_utils.is_sentencepiece_available
    consumer = types.ModuleType("fake_consumer")
    consumer.is_sentencepiece_available = original
    sys.modules["fake_consumer"] = consumer
    try:
        assert getattr(module, patcher)(import_utils) is True
        assert import_utils.is_sentencepiece_available() is False
        assert consumer.is_sentencepiece_available() is False, (
            "a captured copy must be corrected too, or models.auto still tries the "
            "blocked extension"
        )
        assert import_utils.BACKENDS_MAPPING["sentencepiece"][0]() is False
        assert (
            import_utils.BACKENDS_MAPPING["sentencepiece"][1] == "pip install sentencepiece"
        ), "requires_backends must still be able to tell the user how to install it"
    finally:
        sys.modules.pop("fake_consumer", None)


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
def test_a_transformers_4x_layout_is_corrected_through_its_global(
    module, probe, patcher, policy, entry, cache
):
    """The older shape, still what a pinned 4.57 install has."""
    import_utils = _transformers_4x()
    assert getattr(module, patcher)(import_utils) is True
    assert import_utils.is_sentencepiece_available() is False
    assert import_utils._sentencepiece_available is False


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
def test_a_working_extension_is_left_alone(module, probe, patcher, policy, entry, cache):
    """The 99.9% case on the platform this runs on. A healthy Windows machine must lose
    nothing, and must not even read the registry."""
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.delitem(sys.modules, "sentencepiece", raising = False)
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
        monkeypatch.setattr(importlib, "import_module", lambda name, *a, **k: object())

        def _no_registry():
            raise AssertionError("a healthy machine must not be asked about policy")

        monkeypatch.setattr(module, policy, _no_registry)
        with warnings.catch_warnings(record = True) as caught:
            warnings.simplefilter("always")
            assert getattr(module, entry)() is False
        assert caught == []
    finally:
        monkeypatch.undo()


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
def test_nothing_is_imported_off_windows(module, probe, patcher, policy, entry, cache):
    """The probe is an import; paying it on Linux and macOS would slow every launch to
    answer a question already answered."""
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(
            importlib,
            "import_module",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("imported off Windows")),
        )
        assert getattr(module, probe)() is None
        assert getattr(module, entry)() is False
    finally:
        monkeypatch.undo()


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
def test_a_block_warns_and_names_the_loader_error(
    module, probe, patcher, policy, entry, cache, monkeypatch
):
    _blocked_import(monkeypatch)
    import_utils = _transformers_5x()
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    monkeypatch.setitem(sys.modules, "transformers.utils", types.ModuleType("transformers.utils"))
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", import_utils)

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert getattr(module, entry)() is True

    assert import_utils.is_sentencepiece_available() is False
    assert "577" in str(caught[0].message)


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
def test_the_policy_state_is_not_what_decides(
    module, probe, patcher, policy, entry, cache, monkeypatch
):
    """Smart App Control blocks by reputation, one file at a time. It is on for many
    machines whose sentencepiece loads perfectly and off on machines where an antivirus
    refuses the same file, so deciding on the state would break the first group and miss
    the second."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "sentencepiece", raising = False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(importlib, "import_module", lambda name, *a, **k: object())
    monkeypatch.setattr(module, policy, lambda: 1)
    assert getattr(module, entry)() is False


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
def test_a_denied_policy_read_is_unknown_not_off(
    module, probe, patcher, policy, entry, cache, monkeypatch
):
    """It reads HKLM, which a standard user can read, so it needs no elevation and
    prompts for nothing. Every "cannot say" is None."""
    assert getattr(module, policy)() is None  # not Windows, here

    monkeypatch.setattr(sys, "platform", "win32")
    fake = types.ModuleType("winreg")
    fake.HKEY_LOCAL_MACHINE = 0
    fake.KEY_READ = 0
    fake.OpenKey = lambda *a, **k: (_ for _ in ()).throw(PermissionError("denied"))
    fake.QueryValueEx = lambda *a, **k: (0, 0)
    monkeypatch.setitem(sys.modules, "winreg", fake)
    assert getattr(module, policy)() is None


def test_the_studio_copy_does_not_import_unsloth():
    """The reason there are two. Importing unsloth here runs unsloth/__init__.py, whose
    GPU branch pulls torch, Triton, transformers and the model stack into the parent
    process, and can open a competing GPU context on a machine already short of memory."""
    source = (REPO / "studio" / "backend" / "utils" / "sentencepiece_guard.py").read_text(
        encoding = "utf-8"
    )
    assert "import unsloth" not in source
    assert "from unsloth" not in source
    for heavy in ("import torch", "import transformers\n", "import numpy"):
        assert heavy not in source, f"the Studio parent must not pay for {heavy!r}"

    main = (REPO / "studio" / "backend" / "main.py").read_text(encoding = "utf-8")
    assert "from utils.sentencepiece_guard import disable_sentencepiece_if_blocked" in main
    assert "from unsloth.import_fixes import" not in main
