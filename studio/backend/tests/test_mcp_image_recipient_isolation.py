# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from core.inference import mcp_client


@pytest.fixture
def recipient(monkeypatch):
    closes = []
    transport = SimpleNamespace(
        account = "owner",
        location = "https://private-recipient.example/mcp",
        created_at = time.monotonic(),
        close = lambda: closes.append("closed"),
    )
    monkeypatch.setattr(mcp_client, "_private_recipients", {"opaque": transport})
    monkeypatch.setattr(mcp_client, "_private_recipients_lock", threading.Lock())
    return transport, closes


def test_other_account_cannot_inspect_or_close_private_recipient(recipient, monkeypatch):
    transport, closes = recipient
    monkeypatch.setattr(mcp_client, "current_account_id", lambda: "other")
    for operation in (
        mcp_client.mcp_image_recipient_location,
        mcp_client.mcp_image_recipient_remaining_ms,
    ):
        with pytest.raises(mcp_client._PrivateTransportUnavailable):
            operation("opaque")
    mcp_client.close_mcp_image_recipient("opaque")
    assert mcp_client._private_recipients == {"opaque": transport}
    assert closes == []
    monkeypatch.setattr(mcp_client, "current_account_id", lambda: "owner")
    assert mcp_client.mcp_image_recipient_location("opaque") == transport.location
    assert 0 < mcp_client.mcp_image_recipient_remaining_ms("opaque") <= 300_000


def test_owner_concurrent_closes_claim_private_recipient_once(recipient, monkeypatch):
    _, closes = recipient
    monkeypatch.setattr(mcp_client, "current_account_id", lambda: "owner")
    barrier = threading.Barrier(2)

    def close():
        barrier.wait()
        mcp_client.close_mcp_image_recipient("opaque")

    with ThreadPoolExecutor(max_workers = 2) as executor:
        list(executor.map(lambda _: close(), range(2)))
    assert closes == ["closed"]
    assert mcp_client._private_recipients == {}
    mcp_client.close_mcp_image_recipient("opaque")
    for operation in (
        mcp_client.mcp_image_recipient_location,
        mcp_client.mcp_image_recipient_remaining_ms,
    ):
        with pytest.raises(mcp_client._PrivateTransportUnavailable):
            operation("opaque")
    assert closes == ["closed"]
