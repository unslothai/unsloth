# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tell transformers sentencepiece is absent on Windows, or when its extension is refused.

Standalone on purpose. The same guard exists in ``unsloth.import_fixes`` for people who
installed the pip package, but importing it from here would run ``unsloth/__init__.py``,
whose GPU branch pulls in torch, Triton, transformers and the model stack. The Studio
parent is deliberately light because the ML work happens in spawned workers, so that
would add a full stack to the long-lived process and can open a competing GPU context
on the machines least able to afford one. Nothing imported here is heavy. Where the
correction applies, ``transformers.utils.import_utils`` is imported so the flag can be
set before any tokenizer is built; that is transformers' pure-python availability module
and pulls in no torch, and it is what makes the correction land at all.

The two copies must agree; ``tests/studio/test_sentencepiece_guard_parity.py`` holds
them to the same error codes and the same decision.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
import warnings
from typing import Optional

logger = logging.getLogger(__name__)

# Windows loader errors that mean "this image was refused", not "this file is broken".
# 577 ERROR_INVALID_IMAGE_HASH is Smart App Control and App Control for Business, 225
# ERROR_VIRUS_INFECTED is an antivirus blocking on access, 1260
# ERROR_ACCESS_DISABLED_BY_POLICY is AppLocker or SRP. They are separated only so the
# message can name a cause; every failure to load the extension is handled the same way.
BLOCKED_IMAGE_WINERRORS = frozenset({225, 577, 1260})

_RESULT: Optional[bool] = None

# On Windows the extension is disabled by DEFAULT, not only when its load is refused. A
# deliberate temporary measure: the refusals arrive faster than they can be diagnosed one
# machine at a time, they are silent here in particular (a substituted chat template, not
# an error), and the cost of going without is now small. transformers 4.57.6 gates 67
# tokenizer entries on sentencepiece and 5.x gates 8; across the 93 models the Unsloth
# notebooks use, 3 break without it on 4.57.6 and none on 5.5.0 or 5.10.4.
#
# The package stays installed and installable. This flag is the way back.
DISABLE_SENTENCEPIECE_VARIABLE = "UNSLOTH_DISABLE_SENTENCEPIECE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def sentencepiece_disabled_by_policy() -> Optional[str]:
    """Why policy disables sentencepiece here, or None to leave it to the import probe.

    ``"environment"`` when ``UNSLOTH_DISABLE_SENTENCEPIECE`` is truthy, which applies on
    every platform, and ``"windows"`` for the platform default. Off Windows there is no
    default, and WSL reports ``linux``, so a WSL session is treated as the Linux box it is
    rather than inheriting a Windows policy that has no loader to justify it.

    An unrecognised value falls back to the platform default rather than raising. This
    runs at the top of the Studio process, where a typo in an environment variable must
    not cost the user the backend.

    A falsy value only says policy is not disabling it. It cannot re-enable an extension
    the loader refuses, because the caller runs the block detection afterwards either way.
    """
    value = (os.environ.get(DISABLE_SENTENCEPIECE_VARIABLE) or "").strip().lower()
    if value in _TRUTHY:
        return "environment"
    if value in _FALSY:
        return None
    return "windows" if sys.platform == "win32" else None


def smart_app_control_state() -> Optional[int]:
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


def sentencepiece_import_error() -> Optional[BaseException]:
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


def _materialise_tokenizer_mapping(import_utils) -> None:
    """Import transformers' auto tokenizer module before the flag is changed.

    Ordering is load bearing, and the intuitive order is the wrong one. On 4.52 through
    4.57, ``models/auto/tokenization_auto.py`` evaluates ``is_sentencepiece_available()``
    while building ``TOKENIZER_MAPPING_NAMES`` (71 times in 4.57.6). Patch first and the
    slow entries are built as ``None``; ``tokenizer_class_from_name`` then misses the
    mapping and falls through to the dummy-class fallback, which does
    ``importlib.import_module("transformers")`` and reaches an unguarded top level
    ``import sentencepiece as spm`` in the slow tokenizer module. The user gets the raw
    loader error the guard exists to prevent. Measured on 4.57.6 with the extension
    blocked: no patch fails, patching first fails identically, patching after this import
    loads the tokenizer.

    On 5.x the mapping gates only 8 model types and every ordering works, so doing it
    unconditionally costs nothing and keeps one code path.

    Only for the real transformers module. The synthetic modules the parity tests pass in
    must not drag the real package into the process, or the tests stop testing the shapes
    they claim to.
    """
    if not getattr(import_utils, "__name__", "").startswith("transformers"):
        return
    try:
        importlib.import_module("transformers.models.auto.tokenization_auto")
    except Exception:
        # A transformers too broken to import its own auto module is not something to
        # fail on here; the caller still patches, and the flag is still corrected.
        pass


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

    # Before anything is rebound. See the helper: on 4.x the mapping must already be
    # built, or the correction sends the user into the loader error instead of past it.
    _materialise_tokenizer_mapping(import_utils)

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
            if not isinstance(entry, tuple) or not entry:
                continue
            # By key as well as by identity. Identity alone misses an entry transformers
            # built from a different callable, and key alone would rewrite an unrelated
            # backend, so either match is enough but the name is checked too.
            if (
                entry[0] is original
                or str(key).split(">")[0].split("=")[0].strip() == "sentencepiece"
            ):
                mapping[key] = (_sentencepiece_is_absent,) + tuple(entry[1:])

    try:
        return import_utils.is_sentencepiece_available() is False
    except Exception:
        return False


def disable_sentencepiece_if_blocked() -> bool:
    """Correct the availability flag when policy or the loader says sentencepiece is out.

    Two triggers, checked in that order: ``sentencepiece_disabled_by_policy`` for the
    Windows default and the explicit flag, then the import probe for a machine whose
    extension is genuinely refused.

    Returns True only when sentencepiece was actually turned off here. On every other
    machine this is a no-op: policy is silent and the extension imported. The verdict is
    cached, except when transformers is not imported yet, so a later call still gets its
    chance.
    """
    global _RESULT
    if _RESULT is not None:
        return _RESULT

    # Policy first, and the probe strictly as a fallback. The Windows default exists so
    # the extension is never touched, so probing first would hand the blocked file exactly
    # the load the default is there to avoid.
    policy = sentencepiece_disabled_by_policy()
    exception = sentencepiece_import_error() if policy is None else None
    if policy is None and exception is None:
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

    if exception is None:
        # Logged, not warned. A block is a fault on one machine that the operator has to
        # act on, so it earns a warning. The policy default is the expected state of every
        # healthy Windows box, and warnings.warn prints by default: a UserWarning on every
        # single launch is the kind of noise that teaches people to stop reading them. The
        # models that need the extension still fail loudly at the point of use with the
        # backend message naming it, which is where the question is actually asked.
        if policy == "environment":
            logger.info(
                "Unsloth: sentencepiece is disabled because %s is set. transformers will "
                "use fast tokenizers only.",
                DISABLE_SENTENCEPIECE_VARIABLE,
            )
        else:
            logger.info(
                "Unsloth: sentencepiece is disabled by default on Windows, so transformers "
                "will use fast tokenizers only. Nothing was blocked and nothing was "
                "uninstalled; this is a temporary measure while the Windows loader "
                "refusals are worked through. Set %s=0 to use it again.",
                DISABLE_SENTENCEPIECE_VARIABLE,
            )
        _RESULT = True
        return True

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
