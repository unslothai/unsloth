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

# path -> (period_seconds, provenance)
#
# Polls that run when NOTHING is happening: the app is open and the user is not doing
# anything. This is the scenario the idle envelope is set against, and the one a new chatty
# endpoint will usually land in.
IDLE_POLLS: dict[str, tuple[float, str]] = {
    # The loaded-models indicator fires all four together for as long as the app is open.
    # This burst is what the shared liveness bucket exists to collapse.
    "/api/auth/status": (5.0, "measured"),
    "/api/inference/monitor": (5.0, "measured"),
    "/api/inference/images/status": (5.0, "measured"),
    "/api/inference/video/status": (5.0, "measured"),
    "/api/inference/audio/stt/status": (5.0, "measured"),
    # Deliberately outside the shared bucket: their own latency is worth seeing, because
    # /api/health waits on hardware detection and /api/inference/status reads llama.cpp
    # capabilities.
    "/api/health": (5.0, "measured"),
    "/api/inference/status": (5.0, "measured"),
    # The desktop shell's watchdog probe: HEALTH_WATCHDOG_INTERVAL between rounds.
    "/api/liveness": (15.0, "measured"),
    # Re-read while the settings dialog or the remote-access section is open.
    "/api/settings/remote-access": (5.0, "measured"),
    # Tab lists, refetched on a timer and on every tab switch.
    "/api/train/runs": (20.0, "measured"),
    "/api/models/checkpoints": (20.0, "measured"),
    "/api/models/local": (20.0, "measured"),
    "/api/rag/knowledge-bases": (20.0, "measured"),
    # Chat lists, polled while the sidebar is open.
    "/api/chat/projects": (10.0, "declared"),
    "/api/chat/threads": (10.0, "declared"),
    # Small reads the shell and the settings pane make on a slow timer.
    "/api/llama/update-status": (5.0, "declared"),
    "/api/models/loras": (10.0, "declared"),
    "/api/providers/": (10.0, "declared"),
    "/api/providers/registry": (10.0, "declared"),
    "/api/settings/personalization": (10.0, "declared"),
    "/api/system": (10.0, "declared"),
}

# Polls that only exist while an operation is in flight: a download running, a generation
# running, a training run live, the log viewer open. Budgeted separately because holding
# all of them at once is not idle, and averaging them into the idle envelope would hide a
# regression in either direction.
BUSY_POLLS: dict[str, tuple[float, str]] = {
    "/api/models/download-progress": (1.0, "declared"),
    "/api/models/gguf-download-progress": (1.0, "declared"),
    "/api/datasets/download-progress": (1.0, "declared"),
    "/api/inference/images/generate-progress": (0.3, "declared"),
    "/api/inference/video/generate-progress": (0.3, "declared"),
    "/api/inference/images/load-progress": (1.0, "declared"),
    "/api/inference/video/load-progress": (1.0, "declared"),
    "/api/train/diffusion/status": (1.5, "declared"),
    # Suppressed entirely on success, but still polled, and still logged on failure.
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
    # The log viewer reading its own log. Suppressed so that watching a log cannot append
    # to the log being watched.
    "/api/settings/debug/logs": (3.0, "declared"),
    "/api/settings/debug/logs/sources": (3.0, "declared"),
    # Training panels, live only while a run is going.
    "/api/train/status": (2.0, "declared"),
    "/api/train/metrics": (2.0, "declared"),
    "/api/train/hardware": (2.0, "declared"),
}

# Every classified path lives in exactly one scenario. `test_log_budget` proves it.
ALL_POLLS: dict[str, tuple[float, str]] = {**IDLE_POLLS, **BUSY_POLLS}

STEADY_IDLE_SECONDS = 30 * 60
BUSY_SECONDS = 5 * 60

# Paths that are polled often and still sit in the `normal` burst class, so they write a
# line per poll. Introducing this guard to a codebase that already had violations means
# either recording them or weakening the rule, and weakening it would defeat the point.
#
# This list is self-expiring: `test_every_polled_path_has_exactly_one_class` fails if an
# entry here is NO LONGER a violation, so fixing one forces its removal and the list cannot
# quietly become permanent. Do not add to it to make a new endpoint pass.
#
#   /api/settings/remote-access  360 lines per idle half hour  -> fixed by #8763
#   /api/liveness                120 lines per idle half hour  -> fixed by #8763
# Empty, and worth keeping empty. #8763 gave both former entries a heartbeat class, and the
# closure test fails on a stale entry as loudly as on a missing one, so this cannot quietly
# become a place to park a chatty endpoint.
KNOWN_UNCLASSIFIED_POLLS: frozenset[str] = frozenset()

# The envelopes. These are what catch a NEW chatty endpoint: a path can satisfy its own
# class formula perfectly and still push the total up, which is exactly what happened to
# /api/liveness, whose 15s probe sat in the 300ms burst class and logged every single time.
#
# Raising either of these is a product decision about how much Unsloth is allowed to write.
# It is not a knob to turn because a test went red. The class formulas tell you whether the
# suppression rules are being honoured; these tell you whether the result is acceptable.
#
# Set from the measured behaviour of this revision plus room for a genuinely new endpoint,
# NOT from an aspiration.
#
# Ratcheted once already. Before #8763 idle measured 1380 and the envelope was 1450; the
# note here predicted the fix would take idle to roughly 930. It did not: idle now measures
# 1110. The prediction assumed /api/settings/remote-access would stop contributing, but a
# 10s heartbeat against a 5s poll still emits half of them, 180 lines over 30 minutes,
# which is now the joint-largest contributor. Left at 1450 the envelope would have had 340
# lines of slack, which is room for a whole new chatty endpoint to arrive unnoticed.
#
# Re-measure and ratchet again whenever a suppression rule changes. An envelope carrying
# the old number after a fix has stopped guarding anything.
STEADY_IDLE_LINE_ENVELOPE = 1170
# Media milestones reduce this five-minute replay from 239 to 119 lines.
BUSY_LINE_ENVELOPE = 130

# One-shot requests the app makes once on startup. Present so the boot window is not
# mistaken for steady state, and so a mutation record and a failure record exist to assert
# against.
BOOT_REQUESTS: tuple[tuple[str, str, int], ...] = (
    ("POST", "/api/auth/login", 200),
    ("GET", "/api/settings", 200),
    ("GET", "/api/models/list", 200),
    ("GET", "/api/chat/threads", 200),
    ("GET", "/api/inference/status", 401),
)
