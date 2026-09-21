# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""load_model_defaults must not raise on a None/empty model id.

Before the guard, calling it before a model is selected logged
`Error loading model defaults for None: 'NoneType' object has no attribute 'lower'`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils.models.model_config import load_model_defaults  # noqa: E402


def test_none_and_empty_return_empty_without_error(caplog):
    with caplog.at_level(logging.ERROR):
        assert load_model_defaults(None) == {}  # type: ignore[arg-type]
        assert load_model_defaults("") == {}
    assert "Error loading model defaults" not in caplog.text
    assert "NoneType" not in caplog.text


def test_unknown_string_still_returns_defaults_dict():
    # A non-None unknown model name still resolves (falls back to default.yaml),
    # i.e. the guard only short-circuits None/empty, nothing else.
    result = load_model_defaults("definitely-not-a-real-model-xyz")
    assert isinstance(result, dict)
