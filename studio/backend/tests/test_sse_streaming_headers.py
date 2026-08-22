# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the shared SSE streaming-response helper.

Streaming endpoints must disable nginx buffering (``X-Accel-Buffering: no``).
Cloudflare Quick Tunnels also require first-party clients to use POST because
they buffer streamed GET responses. The native ``/generate/stream`` and legacy
``/v1/completions`` streams route through this shared header helper.
"""

import routes.inference as inference_route


def test_sse_helper_sets_no_proxy_buffering_headers():
    resp = inference_route._sse_streaming_response(iter(()))
    assert resp.media_type == "text/event-stream"
    # Starlette lowercases header keys in init_headers.
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["connection"] == "close"
    assert resp.headers["x-accel-buffering"] == "no"
