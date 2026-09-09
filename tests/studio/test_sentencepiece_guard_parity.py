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
def _reset(monkeypatch):
    PACKAGE_COPY._SENTENCEPIECE_GUARD_RESULT = None
    STUDIO_COPY._RESULT = None
    # The flag decides half of what is under test here, so an operator who exported it in
    # the shell running pytest must not be able to change what these tests mean.
    monkeypatch.delenv(VARIABLE, raising = False)
    yield
    PACKAGE_COPY._SENTENCEPIECE_GUARD_RESULT = None
    STUDIO_COPY._RESULT = None


def _variable(module):
    """The env var name, under whichever of the two spellings this copy uses."""
    return getattr(module, "DISABLE_SENTENCEPIECE_VARIABLE", None) or getattr(
        module, "_DISABLE_SENTENCEPIECE_VARIABLE"
    )


VARIABLE = _variable(PACKAGE_COPY)


def _install_fake_transformers(monkeypatch, import_utils):
    """A transformers whose import_utils is the fake, so the entry point patches that."""
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    monkeypatch.setitem(sys.modules, "transformers.utils", types.ModuleType("transformers.utils"))
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", import_utils)


def _refuse_every_import(monkeypatch, reason):
    """Any import at all becomes the test failure it would be in production."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    def _explode(name, *args, **kwargs):
        raise AssertionError(reason)

    monkeypatch.setattr(importlib, "import_module", _explode)


def _blocked_import(
    monkeypatch,
    winerror = 577,
    flag = "0",
):
    """A Windows box whose sentencepiece is installed and refuses to load.

    The flag defaults to "0" because the block path is only reachable that way now: with
    the flag unset, Windows disables sentencepiece by policy and never probes it. Blocked
    machines are therefore covered by the policy default first, and this path is what
    still protects the user who turned the policy back off.
    """
    monkeypatch.setenv(VARIABLE, flag)
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
    """A healthy Windows machine that has opted back in must lose nothing, and must not
    even read the registry. The opt-in is what this test is about: with the flag unset,
    Windows now disables sentencepiece by policy whether or not it would have loaded."""
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setenv(VARIABLE, "0")
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
    the second. Asked with the flag off, since that is the only configuration in which the
    import probe still runs at all."""
    monkeypatch.setenv(VARIABLE, "0")
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


# ---- the policy default -------------------------------------------------------------
#
# Windows disables sentencepiece by default, not only when a block is detected. A
# temporary measure, so the flag that turns it back on is part of the contract and is
# tested as such.


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
def test_both_copies_name_the_same_variable_and_helper(
    module, probe, patcher, policy, entry, cache
):
    """Drift here would mean one copy honours a flag the other has never heard of."""
    assert _variable(module) == "UNSLOTH_DISABLE_SENTENCEPIECE"
    assert callable(module.sentencepiece_disabled_by_policy)


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
def test_windows_disables_by_default_without_touching_the_extension(
    module, probe, patcher, policy, entry, cache, monkeypatch
):
    """The whole point of the default: on Windows the extension is never loaded, so a
    machine whose loader would refuse it never gives the loader the chance. Any import at
    all here is the failure."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "sentencepiece", raising = False)
    _refuse_every_import(monkeypatch, "the policy default must not import sentencepiece")
    import_utils = _transformers_5x()
    _install_fake_transformers(monkeypatch, import_utils)

    assert module.sentencepiece_disabled_by_policy() == "windows"
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert getattr(module, entry)() is True
    assert import_utils.is_sentencepiece_available() is False
    assert caught == [], (
        "a healthy Windows machine gets this on every launch, so it is logged rather "
        "than warned; only a real block earns a warning"
    )


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
@pytest.mark.parametrize("shape", ["4x", "5x"])
def test_the_policy_path_corrects_both_transformers_shapes(
    module, probe, patcher, policy, entry, cache, shape, monkeypatch
):
    """The policy default reaches the same correction as the block path, on the 4.57.6
    module global and on the 5.x lru_cache function alike."""
    monkeypatch.setattr(sys, "platform", "win32")
    import_utils = _transformers_4x() if shape == "4x" else _transformers_5x()
    _install_fake_transformers(monkeypatch, import_utils)

    assert getattr(module, entry)() is True
    assert import_utils.is_sentencepiece_available() is False


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_no_platform_default_off_windows(
    module, probe, patcher, policy, entry, cache, platform, monkeypatch
):
    """Linux and macOS keep sentencepiece. WSL reports "linux", so it lands here too,
    which is right: it runs the Linux loader, not the Windows one."""
    monkeypatch.setattr(sys, "platform", platform)
    import_utils = _transformers_5x()
    _install_fake_transformers(monkeypatch, import_utils)

    assert module.sentencepiece_disabled_by_policy() is None
    assert getattr(module, entry)() is False
    assert import_utils.is_sentencepiece_available() is True


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
@pytest.mark.parametrize("value", ["1", "true", "YES", " On ", "TRUE"])
def test_a_truthy_flag_disables_on_every_platform(
    module, probe, patcher, policy, entry, cache, value, monkeypatch
):
    """Case and surrounding whitespace are the operator's, not the contract's."""
    monkeypatch.setenv(VARIABLE, value)
    monkeypatch.setattr(sys, "platform", "linux")
    import_utils = _transformers_5x()
    _install_fake_transformers(monkeypatch, import_utils)

    assert module.sentencepiece_disabled_by_policy() == "environment"
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert getattr(module, entry)() is True
    assert import_utils.is_sentencepiece_available() is False
    assert caught == [], "the operator asked for this; telling them so is pure noise"


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
@pytest.mark.parametrize("value", ["0", "false", "NO", " off "])
def test_a_falsy_flag_on_windows_leaves_a_healthy_machine_alone(
    module, probe, patcher, policy, entry, cache, value, monkeypatch
):
    """The way back. An extension that loads stays available."""
    monkeypatch.setenv(VARIABLE, value)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delitem(sys.modules, "sentencepiece", raising = False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(importlib, "import_module", lambda name, *a, **k: object())
    import_utils = _transformers_5x()
    _install_fake_transformers(monkeypatch, import_utils)

    assert module.sentencepiece_disabled_by_policy() is None
    assert getattr(module, entry)() is False
    assert import_utils.is_sentencepiece_available() is True


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
def test_a_falsy_flag_cannot_re_enable_a_refused_extension(
    module, probe, patcher, policy, entry, cache, monkeypatch
):
    """The important half of the flag. It says "do not disable this by policy", not "this
    works": a user who opts back in on a machine whose loader refuses the file is still
    protected, and still gets the warning naming the loader error."""
    _blocked_import(monkeypatch, flag = "0")
    import_utils = _transformers_5x()
    _install_fake_transformers(monkeypatch, import_utils)

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        assert getattr(module, entry)() is True
    assert import_utils.is_sentencepiece_available() is False
    assert "577" in str(caught[0].message)


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
@pytest.mark.parametrize("value", ["maybe", "2", "", "   ", "disable"])
def test_an_unrecognised_value_falls_back_to_the_platform_default(
    module, probe, patcher, policy, entry, cache, value, monkeypatch
):
    """This runs during import. Raising on a typo would cost the user the whole package
    over a shell variable, so an unreadable value is treated as unset."""
    monkeypatch.setenv(VARIABLE, value)
    monkeypatch.setattr(sys, "platform", "win32")
    assert module.sentencepiece_disabled_by_policy() == "windows"
    monkeypatch.setattr(sys, "platform", "linux")
    assert module.sentencepiece_disabled_by_policy() is None


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
def test_a_policy_disable_before_transformers_does_not_poison_the_cache(
    module, probe, patcher, policy, entry, cache, monkeypatch
):
    """Same rule as the block path: a call that could not reach transformers has not
    decided anything, so the next call must get its chance."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "transformers", None)
    assert getattr(module, entry)() is False
    assert getattr(module, cache) is None

    monkeypatch.delitem(sys.modules, "transformers")
    import_utils = _transformers_5x()
    _install_fake_transformers(monkeypatch, import_utils)
    assert getattr(module, entry)() is True
    assert getattr(module, cache) is True


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


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
def test_the_auto_tokenizer_mapping_is_built_before_the_flag_is_changed(
    module, probe, patcher, policy, entry, cache, monkeypatch
):
    """Ordering, and the intuitive order is the wrong one.

    transformers 4.52 through 4.57 evaluate is_sentencepiece_available() while building
    TOKENIZER_MAPPING_NAMES. Correct the flag first and the slow entries are built as
    None, tokenizer_class_from_name misses the mapping, and the dummy-class fallback
    imports transformers and reaches an unguarded top level `import sentencepiece as spm`.
    The user gets the raw loader error this guard exists to prevent. Measured on 4.57.6
    with the extension blocked: unpatched fails, patched-first fails identically, and
    patched after this import loads the tokenizer.

    Asserting only that is_sentencepiece_available() ends up False would pass in both
    orderings and catch none of this, so the order itself is what is pinned.
    """
    order = []
    import_utils = _transformers_4x()
    # Named like the real module, since the helper deliberately only does this for
    # transformers and ignores the synthetic modules the other cases pass in.
    import_utils.__name__ = "transformers.utils.import_utils"
    original = import_utils.is_sentencepiece_available

    def _record_import(name, *args, **kwargs):
        order.append(("import", name))
        return types.ModuleType(name)

    monkeypatch.setattr(importlib, "import_module", _record_import)

    class _Watched(dict):
        def __setitem__(self, key, value):
            order.append(("mapping", key))
            super().__setitem__(key, value)

    import_utils.BACKENDS_MAPPING = _Watched(
        {"sentencepiece": (original, "pip install sentencepiece")}
    )
    assert getattr(module, patcher)(import_utils) is True

    imports = [name for kind, name in order if kind == "import"]
    assert "transformers.models.auto.tokenization_auto" in imports, (
        "the auto tokenizer module must be imported so its mapping is materialised while "
        "sentencepiece still reads as available"
    )
    first_write = next((i for i, (kind, _) in enumerate(order) if kind == "mapping"), len(order))
    first_import = order.index(("import", "transformers.models.auto.tokenization_auto"))
    assert first_import < first_write, (
        "the mapping must be built BEFORE anything is rebound, or the correction sends "
        "the user into the loader error instead of past it"
    )


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
def test_a_synthetic_module_does_not_drag_in_real_transformers(
    module, probe, patcher, policy, entry, cache, monkeypatch
):
    """The other cases pass hand-built modules. If the helper imported transformers for
    those too, every test here would be running against the installed package instead of
    the shape it claims to model."""
    calls = []
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, *a, **k: (calls.append(name), types.ModuleType(name))[1],
    )
    import_utils = _transformers_5x()  # __name__ is "fake_import_utils"
    assert getattr(module, patcher)(import_utils) is True
    assert calls == [], f"nothing should have been imported, got {calls}"


@pytest.mark.parametrize("module,probe,patcher,policy,entry,cache", COPIES)
def test_a_versioned_backend_entry_is_corrected_too(module, probe, patcher, policy, entry, cache):
    """BACKENDS_MAPPING is matched by key as well as by function identity.

    Latent today: every shipped `@requires(backends=("sentencepiece",))` uses the bare
    name, so identity matching happens to be enough. It stops being enough the moment
    transformers declares a versioned sentencepiece backend, and matching the name costs
    nothing now.
    """
    import_utils = _transformers_5x()
    other = lambda: True  # noqa: E731 - a distinct object, so identity cannot match
    import_utils.BACKENDS_MAPPING["sentencepiece>=0.1.91"] = (other, "pip install sentencepiece")
    assert getattr(module, patcher)(import_utils) is True
    assert import_utils.BACKENDS_MAPPING["sentencepiece>=0.1.91"][0]() is False
