# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What an idle Unsloth actually asks for, and how often.

Every path the middleware classifies has to appear here with a poll period. That is the
whole mechanism: you cannot quiet a path without saying how often it is polled, and you
cannot start polling a path without classifying it. ``test_log_budget`` checks both
directions, so this file and ``loggers/handlers.py`` cannot drift apart silently.

Periods marked "measured" come from driving two real Unsloth instances for twelve minutes
and reading the access log back. The rest are the interval the UI declares at its call
site, and are marked "declared"; they are used identically by the replay and by the
expectation, so an imprecise one costs realism in the global envelope, never correctness of
a per-class check.
"""

from __future__ import annotations

# path -> (period_seconds, provenance). Polls while the app is open and idle.
IDLE_POLLS: dict[str, tuple[float, str]] = {
    # The loaded-models indicator fires these together; the shared liveness bucket collapses them.
    "/api/auth/status": (5.0, "measured"),
    "/api/inference/monitor": (5.0, "measured"),
    "/api/inference/images/status": (5.0, "measured"),
    "/api/inference/video/status": (5.0, "measured"),
    "/api/inference/audio/stt/status": (5.0, "measured"),
    # Outside the shared bucket on purpose: their latency is worth seeing.
    "/api/health": (5.0, "measured"),
    "/api/inference/status": (5.0, "measured"),
    "/api/liveness": (15.0, "measured"),
    "/api/settings/remote-access": (5.0, "measured"),
    "/api/train/runs": (20.0, "measured"),
    "/api/models/checkpoints": (20.0, "measured"),
    "/api/models/local": (20.0, "measured"),
    "/api/rag/knowledge-bases": (20.0, "measured"),
    "/api/chat/projects": (10.0, "declared"),
    "/api/chat/threads": (10.0, "declared"),
    "/api/llama/update-status": (5.0, "declared"),
    "/api/models/loras": (10.0, "declared"),
    "/api/providers/": (10.0, "declared"),
    "/api/providers/registry": (10.0, "declared"),
    "/api/settings/personalization": (10.0, "declared"),
    "/api/system": (10.0, "declared"),
}

# Polls only while an operation is in flight; separate so the idle envelope is not averaged.
BUSY_POLLS: dict[str, tuple[float, str]] = {
    "/api/models/download-progress": (1.0, "declared"),
    "/api/models/gguf-download-progress": (1.0, "declared"),
    "/api/datasets/download-progress": (1.0, "declared"),
    "/api/inference/images/generate-progress": (0.3, "declared"),
    "/api/inference/video/generate-progress": (0.3, "declared"),
    "/api/inference/images/load-progress": (1.0, "declared"),
    "/api/inference/video/load-progress": (1.0, "declared"),
    "/api/train/diffusion/status": (1.5, "declared"),
    "/api/export/logs": (1.0, "declared"),
    "/api/export/status": (1.0, "declared"),
    "/api/hub/active-downloads": (1.0, "declared"),
    "/api/hub/datasets/active-downloads": (1.0, "declared"),
    "/api/hub/datasets/download-progress": (1.0, "declared"),
    "/api/hub/datasets/download-status": (1.0, "declared"),
    "/api/hub/datasets/transport-status": (1.0, "declared"),
    "/api/hub/download-progress": (1.0, "declared"),
    "/api/hub/download-status": (1.0, "declared"),
    "/api/hub/gguf-download-progress": (1.0, "declared"),
    "/api/hub/transport-status": (1.0, "declared"),
    "/api/inference/load-progress": (1.0, "declared"),
    # Suppressed so watching a log cannot append to it.
    "/api/settings/debug/logs": (3.0, "declared"),
    "/api/settings/debug/logs/sources": (3.0, "declared"),
    # Busy-only, driven by streaming persistence; measured ~0.5s apart, rounded to the replay tick.
    "/api/chat/threads/{id}": (0.5, "measured"),
    "/api/chat/threads/{id}/forks": (0.5, "measured"),
    "/api/train/status": (2.0, "declared"),
    "/api/train/metrics": (2.0, "declared"),
    "/api/train/hardware": (2.0, "declared"),
}

ALL_POLLS: dict[str, tuple[float, str]] = {**IDLE_POLLS, **BUSY_POLLS}

STEADY_IDLE_SECONDS = 30 * 60
BUSY_SECONDS = 5 * 60

# Self-expiring: the closure test fails on a stale entry. Do not add to make a new endpoint pass.
KNOWN_UNCLASSIFIED_POLLS: frozenset[str] = frozenset()

# Envelopes catch a NEW chatty endpoint. Raising them is a product decision, not a test fix.
# Set from measured behaviour plus small slack; re-measure when a suppression rule changes.
STEADY_IDLE_LINE_ENVELOPE = 1170
BUSY_LINE_ENVELOPE = 195

BOOT_REQUESTS: tuple[tuple[str, str, int], ...] = (
    ("POST", "/api/auth/login", 200),
    ("GET", "/api/settings", 200),
    ("GET", "/api/models/list", 200),
    ("GET", "/api/chat/threads", 200),
    ("GET", "/api/inference/status", 401),
)
