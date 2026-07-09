# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from hub.services import download_lifecycle


def _set_xet_reason(monkeypatch, reason):
    monkeypatch.setattr(
        download_lifecycle.download_registry,
        "download_transport_unavailable_reason",
        lambda _transport: reason,
    )


def test_resolve_effective_use_xet_keeps_http_when_not_requested(monkeypatch):
    _set_xet_reason(monkeypatch, "should not be consulted")
    assert download_lifecycle.resolve_effective_use_xet(False) is False


def test_resolve_effective_use_xet_keeps_xet_when_available(monkeypatch):
    _set_xet_reason(monkeypatch, None)
    assert download_lifecycle.resolve_effective_use_xet(True) is True


def test_resolve_effective_use_xet_downgrades_when_xet_unavailable(monkeypatch):
    _set_xet_reason(monkeypatch, "Xet transport is unavailable because hf_xet is not installed.")
    assert download_lifecycle.resolve_effective_use_xet(True) is False
