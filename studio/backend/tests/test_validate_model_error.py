# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""routes/inference.py::validate_model surfaces actionable RuntimeError/ValueError
messages (e.g. "llama-server binary not found - run setup.sh") instead of a blank
"Invalid model", while keeping unexpected exceptions generic so internals never
leak to the client.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402

import routes.inference as inf  # noqa: E402
from models.inference import ValidateModelRequest  # noqa: E402


def _provoke(monkeypatch, exc: BaseException, *, native: bool = False) -> HTTPException:
    """Drive validate_model so from_identifier raises ``exc``; return the
    HTTPException it converts that into."""
    monkeypatch.setattr(
        inf,
        "_resolve_model_identifier_for_request",
        lambda request, operation: ("org/repo", "org/repo", native),
    )

    def _raise(**_kwargs):
        raise exc

    monkeypatch.setattr(inf.ModelConfig, "from_identifier", staticmethod(_raise))

    req = ValidateModelRequest(model_path = "org/repo")
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inf.validate_model(req, current_subject = "tester"))
    return excinfo.value


def test_runtime_error_surfaces_actionable_message(monkeypatch):
    err = RuntimeError(
        "llama-server binary not found - cannot load GGUF models. "
        "Run setup.sh to build it, or set LLAMA_SERVER_PATH."
    )
    http = _provoke(monkeypatch, err)
    assert http.status_code == 400
    assert "llama-server binary not found" in http.detail
    assert http.detail != "Invalid model"


def test_value_error_not_supported_is_wrapped(monkeypatch):
    http = _provoke(monkeypatch, ValueError("architecture FooBar is not supported"))
    assert http.status_code == 400
    assert "not supported yet" in http.detail.lower()
    # Original cause is preserved for context.
    assert "FooBar" in http.detail


def test_unexpected_exception_stays_generic(monkeypatch):
    # A non-user-facing exception type must NOT have its message surfaced.
    http = _provoke(monkeypatch, KeyError("secret-internal-detail"))
    assert http.status_code == 400
    assert http.detail == "Invalid model"
    assert "secret-internal-detail" not in http.detail


def test_empty_runtime_error_falls_back_to_generic(monkeypatch):
    # A RuntimeError with no message should not produce an empty 400 detail.
    http = _provoke(monkeypatch, RuntimeError(""))
    assert http.status_code == 400
    assert http.detail == "Invalid model"
