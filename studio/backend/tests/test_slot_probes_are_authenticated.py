# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``/slots`` is not a public endpoint.

``UNSLOTH_DIRECT_STREAM=1`` starts llama-server with ``--api-key``. llama.cpp's key
middleware exempts only ``/health`` and ``/v1/health`` (plus the bundled UI assets) and
answers every other path 401, so an unauthenticated read of ``/slots`` fails and both
helpers swallow it: the occupancy read becomes None -- "cannot tell" -- and the erase
returns zero tokens freed.

That is not a degraded mode, it is the mechanism switched off. ``note_resident`` never
learns what the cache holds, ``_room_for_locked`` falls back to the ledger alone, and a
paused chat's cells are never erased for the waiter they were freed for.
"""

from __future__ import annotations

import json
import urllib.request

from core.inference.llama_stats import erase_llama_slot, fetch_llama_slots


class _Response:
    status = 200

    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _capture(monkeypatch, body: bytes):
    seen: list[urllib.request.Request] = []

    def _urlopen(request, timeout = None):
        # Both helpers must send a Request, not a bare URL, or there is nowhere to put
        # the header at all.
        assert isinstance(request, urllib.request.Request), request
        seen.append(request)
        return _Response(body)

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    return seen


class TestTheReadCarriesTheKey:
    def test_the_authorization_header_is_sent(self, monkeypatch):
        seen = _capture(monkeypatch, json.dumps([]).encode())
        fetch_llama_slots("http://127.0.0.1:8080", headers = {"Authorization": "Bearer secret-key"})
        assert seen[0].get_header("Authorization") == "Bearer secret-key"

    def test_a_load_without_a_key_still_sends_nothing(self, monkeypatch):
        """The ordinary case. `_auth_headers` is None unless --api-key was used."""
        seen = _capture(monkeypatch, json.dumps([]).encode())
        fetch_llama_slots("http://127.0.0.1:8080")
        assert seen[0].get_header("Authorization") is None


class TestTheEraseCarriesTheKey:
    def test_the_authorization_header_is_sent(self, monkeypatch):
        seen = _capture(monkeypatch, json.dumps({"n_erased": 4096}).encode())
        freed = erase_llama_slot(
            "http://127.0.0.1:8080", 2, headers = {"Authorization": "Bearer secret-key"}
        )
        assert freed == 4096
        assert seen[0].get_header("Authorization") == "Bearer secret-key"
        assert seen[0].get_method() == "POST"


class TestTheRouteHandsThemTheBackendsKey:
    def test_every_slot_call_goes_through_the_header_helper(self):
        from pathlib import Path

        import routes.inference as inference

        source = Path(inference.__file__).read_text(encoding = "utf-8")
        assert "def _llama_slot_headers(" in source
        # Every call site, not most of them: one unauthenticated read is enough to make
        # the residency figure None and disable the probe for everybody.
        assert source.count("fetch_llama_slots(") == source.count(
            "fetch_llama_slots(base, headers = _llama_slot_headers(llama_backend))"
        )
        assert source.count("erase_llama_slot(") == source.count(
            "erase_llama_slot(base, slot_id, headers = _llama_slot_headers(llama_backend))"
        )

    def test_the_helper_reads_the_backends_own_bearer(self):
        from routes.inference import _llama_slot_headers

        class _Backend:
            _auth_headers = {"Authorization": "Bearer abc"}

        class _Unkeyed:
            _auth_headers = None

        assert _llama_slot_headers(_Backend()) == {"Authorization": "Bearer abc"}
        assert _llama_slot_headers(_Unkeyed()) == {}
        assert _llama_slot_headers(object()) == {}
