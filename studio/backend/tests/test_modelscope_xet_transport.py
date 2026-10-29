# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ModelScope cannot serve Hugging Face Xet tokens; oversized downloads must not stall there (#15989)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from hub.services import download_lifecycle as dl
from hub.utils import download_registry

_OVERSIZED = 54_400_261_312
_UNDER_THE_CEILING = 49_859_583_136


def _pretend_hf_xet_installed(monkeypatch):
    real = download_registry.importlib.util.find_spec

    def find_spec(name, package = None):
        if name == "hf_xet":
            return object()
        return real(name, package)

    monkeypatch.setattr(download_registry.importlib.util, "find_spec", find_spec)


@pytest.fixture
def modelscope_source(monkeypatch):
    monkeypatch.setattr("utils.hub_settings.active_source", lambda: "modelscope")
    _pretend_hf_xet_installed(monkeypatch)


@pytest.fixture
def huggingface_source(monkeypatch):
    monkeypatch.setattr("utils.hub_settings.active_source", lambda: "huggingface")
    _pretend_hf_xet_installed(monkeypatch)


def test_modelscope_marks_xet_unavailable(modelscope_source):
    reason = download_registry.modelscope_xet_unavailable_reason()
    assert reason is not None
    assert "ModelScope" in reason and "Hugging Face" in reason

    caps = download_registry.get_download_transport_capabilities()
    assert caps.xet.available is False
    assert caps.xet.reason == reason
    assert caps.auto_resolves_to == download_registry.TRANSPORT_HTTP


def test_huggingface_source_keeps_xet(huggingface_source):
    assert download_registry.modelscope_xet_unavailable_reason() is None
    caps = download_registry.get_download_transport_capabilities()
    assert caps.xet.available is True


def test_modelscope_oversized_download_refuses_with_actionable_message(modelscope_source):
    """Do not attempt Xet (405) then die on the HTTPS 50GB ceiling (#15989)."""
    use_xet, reason = dl.resolve_requested_use_xet(
        "auto", True, largest_file_bytes = _OVERSIZED
    )
    assert use_xet is False
    assert "ModelScope" in reason

    with pytest.raises(HTTPException) as raised:
        dl.resolve_transport(use_xet, largest_file_bytes = _OVERSIZED)
    detail = raised.value.detail
    assert raised.value.status_code == 400
    assert "GB" in detail
    assert "ModelScope" in detail


def test_modelscope_small_file_falls_back_to_http(modelscope_source):
    use_xet, _ = dl.resolve_requested_use_xet("xet", True, largest_file_bytes = _UNDER_THE_CEILING)
    assert use_xet is False
    assert (
        dl.resolve_transport(use_xet, largest_file_bytes = _UNDER_THE_CEILING)
        == download_registry.TRANSPORT_HTTP
    )
