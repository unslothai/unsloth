# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tell transformers sentencepiece is absent when Windows will not load its extension.

Standalone on purpose. The same guard exists in ``unsloth.import_fixes`` for people who
installed the pip package, but importing it from here would run ``unsloth/__init__.py``,
whose GPU branch pulls in torch, Triton, transformers and the model stack. The Studio
parent is deliberately light because the ML work happens in spawned workers, so that
would add a full stack to the long-lived process and can open a competing GPU context
on the machines least able to afford one. Nothing imported here is heavy, and on a
machine where the extension loads nothing at all is imported.

The two copies must agree; ``tests/studio/test_sentencepiece_guard_parity.py`` holds
them to the same error codes and the same decision.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import warnings

# Windows loader errors that mean "this image was refused", not "this file is broken".
# 577 ERROR_INVALID_IMAGE_HASH is Smart App Control and App Control for Business, 225
# ERROR_VIRUS_INFECTED is an antivirus blocking on access, 1260
# ERROR_ACCESS_DISABLED_BY_POLICY is AppLocker or SRP. They are separated only so the
# message can name a cause; every failure to load the extension is handled the same way.
BLOCKED_IMAGE_WINERRORS = frozenset({225, 577, 1260})

_RESULT: bool | None = None


def smart_app_control_state() -> int | None:
    """Smart App Control's state: 0 off, 1 enforced, 2 evaluation, or None if unknown.

    Read from ``HKLM\\SYSTEM\\CurrentControlSet\\Control\\CI\\Policy``, which a standard
    user can already read, so this needs no elevation and prompts for nothing.
    ``Win32_DeviceGuard`` answers a related question and does require admin, which is why
    it is not used.

    None covers every "cannot say": not Windows, no winreg, the value absent because the
    build never configured SAC, or the read denied. Callers must treat None as unknown
    rather than off, and nothing here decides anything on this value alone.
    """
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\CI\Policy",
            0,
            winreg.KEY_READ,
        ) as key:
            value, _ = winreg.QueryValueEx(key, "VerifiedAndReputablePolicyState")
    except (OSError, ValueError):
        return None
    return value if isinstance(value, int) else None


def sentencepiece_import_error() -> BaseException | None:
    """The exception importing sentencepiece raises here, or None when it imports.

    Windows only, and only when the package is installed: everywhere else the answer is
    already right and paying an import to confirm it would slow every launch.

    The import is the only authoritative test. Smart App Control blocks by reputation,
    one file at a time, so it can refuse this extension on a machine where it is on and
    everything else loads, and it can be off while an antivirus refuses the same file.
    Reading the policy state and disabling sentencepiece on that alone would take the
    tokenizer away from the large majority of SAC users whose extension loads fine.
    """
    if sys.platform != "win32":
        return None
    if "sentencepiece" in sys.modules:
        return None
    if importlib.util.find_spec("sentencepiece") is None:
        # Genuinely not installed. transformers already agrees.
        return None
    try:
        importlib.import_module("sentencepiece")
    except Exception as exception:
        return exception
    return None


def tell_transformers_sentencepiece_is_absent(import_utils) -> bool:
    """Make every consumer of ``is_sentencepiece_available`` answer False.

    The two shipped shapes need different work, and neither is covered by patching only
    the defining module:

    4.x keeps a module global, ``_sentencepiece_available``, that the function returns.
    Setting it is enough there and is done first, since it also fixes code reading the
    global directly.

    5.x has no such global: the function is ``@lru_cache``-wrapped around
    ``_is_package_available("sentencepiece")``, which asks ``find_spec`` and loads
    nothing, so it answers True while the extension is refused. Assigning the global
    there creates an unused attribute and changes nothing.

    In both, replacing ``import_utils.is_sentencepiece_available`` is not enough on its
    own. ``BACKENDS_MAPPING`` captures the function object at import time, and so does
    every module that did ``from ... import is_sentencepiece_available`` before this ran.
    Those copies keep the original and keep saying True.

    So the original is rebound everywhere it is already bound: every module in
    sys.modules holding it under any name, and the ``BACKENDS_MAPPING`` entry, whose
    tuple is rebuilt rather than mutated so the install message survives. Modules
    imported afterwards need nothing, since they read the replaced name.
    """
    original = getattr(import_utils, "is_sentencepiece_available", None)
    if original is None:
        return False

    if hasattr(import_utils, "_sentencepiece_available"):
        import_utils._sentencepiece_available = False
    try:
        if import_utils.is_sentencepiece_available() is False:
            return True
    except Exception:
        return False

    def _sentencepiece_is_absent(*args, **kwargs):
        return False

    _sentencepiece_is_absent.__name__ = getattr(original, "__name__", "is_sentencepiece_available")
    import_utils.is_sentencepiece_available = _sentencepiece_is_absent

    for module in list(sys.modules.values()):
        if module is None or module is import_utils:
            continue
        try:
            names = vars(module)
        except Exception:
            continue
        for name, value in list(names.items()):
            if value is original:
                try:
                    setattr(module, name, _sentencepiece_is_absent)
                except Exception:
                    pass

    mapping = getattr(import_utils, "BACKENDS_MAPPING", None)
    if isinstance(mapping, dict):
        for key, entry in list(mapping.items()):
            if isinstance(entry, tuple) and entry and entry[0] is original:
                mapping[key] = (_sentencepiece_is_absent,) + tuple(entry[1:])

    try:
        return import_utils.is_sentencepiece_available() is False
    except Exception:
        return False


def disable_sentencepiece_if_blocked() -> bool:
    """Correct the availability flag when the extension will not load.

    Returns True only when a real block was found and the correction took. On every
    other machine this is a no-op: not Windows, not installed, or it imported. The
    verdict is cached, except when transformers is not imported yet, so a later call
    still gets its chance.
    """
    global _RESULT
    if _RESULT is not None:
        return _RESULT

    exception = sentencepiece_import_error()
    if exception is None:
        _RESULT = False
        return False

    try:
        import transformers.utils.import_utils as import_utils
    except Exception:
        return False

    if not tell_transformers_sentencepiece_is_absent(import_utils):
        # The correction did not take, so warning that it did would be worse than
        # saying nothing: the operator would stop looking.
        return False

    winerror = getattr(exception, "winerror", None)
    if winerror in BLOCKED_IMAGE_WINERRORS:
        cause = f"blocked by Windows (error {winerror})"
    else:
        cause = "could not be loaded"
    state = smart_app_control_state()
    if state == 1:
        cause += "; Smart App Control is on"
    elif state == 2:
        cause += "; Smart App Control is in evaluation mode"
    warnings.warn(
        f"Unsloth: the sentencepiece extension {cause}: {exception}\n"
        "Continuing without it. Models that ship a fast tokenizer (tokenizer.json) are "
        "unaffected; a model that only ships tokenizer.model will now say sentencepiece "
        "is required instead of failing later with a loader error.\n"
        "To restore it, allow the file in your security software, or reinstall "
        "sentencepiece so a differently named copy is written.",
        stacklevel = 2,
    )
    _RESULT = True
    return True
