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

# path -> (period_seconds, provenance). Idle: polls that run when nothing is happening.
IDLE_POLLS: dict[str, tuple[float, str]] = {
    # Fired together for as long as the app is open; the shared liveness bucket collapses them.
    "/api/auth/status": (5.0, "measured"),
    "/api/inference/monitor": (5.0, "measured"),
    "/api/inference/images/status": (5.0, "measured"),
    "/api/inference/video/status": (5.0, "measured"),
    "/api/inference/audio/stt/status": (5.0, "measured"),
    # Deliberately outside the shared bucket: their own latency is worth seeing.
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

# Busy: polls that exist only while an operation is in flight. Budgeted separately so
# holding all of them at once cannot hide a regression in the idle envelope.
BUSY_POLLS: dict[str, tuple[float, str]] = {
    "/api/models/download-progress": (1.0, "declared"),
    "/api/models/gguf-download-progress": (1.0, "declared"),
    "/api/datasets/download-progress": (1.0, "declared"),
    "/api/inference/images/generate-progress": (1.0, "declared"),
    "/api/inference/video/generate-progress": (1.0, "declared"),
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
    # Suppressed so that watching a log cannot append to the log being watched.
    "/api/settings/debug/logs": (3.0, "declared"),
    "/api/settings/debug/logs/sources": (3.0, "declared"),
    # Driven by the streaming persistence loop, not a timer. Measured ~0.4-0.6s apart,
    # rounded to the replay's 0.5s tick.
    "/api/chat/threads/{id}": (0.5, "measured"),
    "/api/chat/threads/{id}/forks": (0.5, "measured"),
    "/api/train/status": (2.0, "declared"),
    "/api/train/metrics": (2.0, "declared"),
    "/api/train/hardware": (2.0, "declared"),
}

ALL_POLLS: dict[str, tuple[float, str]] = {**IDLE_POLLS, **BUSY_POLLS}

STEADY_IDLE_SECONDS = 30 * 60
BUSY_SECONDS = 5 * 60

# Recorded violations: polled-often paths still in the `normal` burst class. Self-expiring --
# `test_every_polled_path_has_exactly_one_class` fails on a stale entry as loudly as on a
# missing one, so do NOT add here to make a new chatty endpoint pass.
KNOWN_UNCLASSIFIED_POLLS: frozenset[str] = frozenset()

# The envelopes catch a NEW chatty endpoint: a path can satisfy its own class formula and
# still push the total up (see /api/liveness, a 15s probe in the 300ms burst class).
# Raising either is a product decision about how much Unsloth may write, not a knob to turn
# because a test went red. Set from measured behaviour of this revision plus room for one
# genuinely new endpoint; re-measure and ratchet whenever a suppression rule changes.
STEADY_IDLE_LINE_ENVELOPE = 1170
BUSY_LINE_ENVELOPE = 325

# Startup one-shots, so the boot window is not mistaken for steady state.
BOOT_REQUESTS: tuple[tuple[str, str, int], ...] = (
    ("POST", "/api/auth/login", 200),
    ("GET", "/api/settings", 200),
    ("GET", "/api/models/list", 200),
    ("GET", "/api/chat/threads", 200),
    ("GET", "/api/inference/status", 401),
)
