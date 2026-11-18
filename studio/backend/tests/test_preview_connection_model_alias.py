# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A public /p link must not reach a saved provider connection through the external:: model alias."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_preview_routes import _sig, captured, client  # noqa: E402, F401


def test_external_model_alias_cannot_route_to_saved_connection(client, captured):
    r = client.post(
        f"/p/demorun/v1/chat/completions?k={_sig('demorun')}",
        json = {
            "model": "external::0123456789abcdef::gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    p = captured["payload"]
    assert not p.model.startswith("external::")
    assert p.provider_id is None


def test_plain_preview_model_name_is_kept(client, captured):
    r = client.post(
        f"/p/demorun/v1/chat/completions?k={_sig('demorun')}",
        json = {"model": "preview", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert captured["payload"].model == "preview"
