# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A torchcodec placeholder for the Colab-shaped smoke job, which has no CPU wheel to install.

The stub used to be a bare ``types.ModuleType``, whose ``__spec__`` is None. That is not an
inert difference: ``importlib.util.find_spec`` raises for a module sitting in ``sys.modules``
with no spec, rather than returning None, so every probe of the form

    importlib.util.find_spec("torchcodec") is not None

turns into a ValueError. transformers reaches exactly that probe while importing
``audio_utils`` (``is_torchcodec_available`` -> ``_is_package_available``), which peft pulls in
through ``BloomPreTrainedModel``, so the smoke job died on ``import peft`` with

    ValueError: torchcodec.__spec__ is None

Carrying a real ModuleSpec makes the probe answer instead of raising. It still answers
"not available": the spec exists, then ``importlib.metadata.version`` raises
PackageNotFoundError because no distribution is installed, which is the outcome the stub
wants and the one a machine without the wheel should report.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

NAME = "torchcodec"


def install() -> types.ModuleType:
    """Put the placeholder in ``sys.modules`` unless something real is already there."""
    existing = sys.modules.get(NAME)
    if existing is not None:
        # A real torchcodec, or a stub installed by an earlier step. Either way, leave it:
        # overwriting a genuine module would be the opposite of what this is for.
        return existing
    module = types.ModuleType(NAME)
    # loader=None marks it as a namespace-ish placeholder rather than claiming an importer
    # that would be asked to load it.
    module.__spec__ = importlib.machinery.ModuleSpec(NAME, None)
    sys.modules[NAME] = module
    return module
