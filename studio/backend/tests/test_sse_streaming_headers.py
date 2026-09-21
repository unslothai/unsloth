# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the shared SSE streaming-response helper.

Streaming endpoints must disable nginx buffering (``X-Accel-Buffering: no``); Cloudflare
Quick Tunnels additionally need first-party clients on POST, since they hold a streamed
GET. ``/generate/stream`` and legacy ``/v1/completions`` route through this helper.
"""

import routes.inference as inference_route


def test_sse_helper_sets_no_proxy_buffering_headers():
    resp = inference_route._sse_streaming_response(iter(()))
    assert resp.media_type == "text/event-stream"
    # Starlette lowercases header keys in init_headers.
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["connection"] == "close"
    assert resp.headers["x-accel-buffering"] == "no"
