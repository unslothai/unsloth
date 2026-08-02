// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DownloadJobState } from "./api";

export const POLL_INTERVAL_MS = 500;
export const POLL_BACKOFF_AFTER_MS = 60_000;
export const POLL_BACKOFF_INTERVAL_MS = 1_500;
export const HIDDEN_POLL_INTERVAL_MS = 10_000;
export const POLL_JITTER_MS = 50;
export const PROGRESS_POLL_INTERVAL_MS = 1_000;
export const PROGRESS_POLL_BACKOFF_INTERVAL_MS = 2_000;
export const POLL_REQUEST_TIMEOUT_MS = 15_000;
export const POLL_DEGRADED_AFTER_MS = 30_000;
export const POLL_DEGRADED_MESSAGE =
  "Couldn't update download status. The download may still be running.";
export const TRANSPORT_STATUS_TIMEOUT_MS = 3_000;
export const SPEED_EMA_WEIGHT = 0.7;
export const MAX_PROGRESS_FRACTION = 0.99;
export const CANCEL_WATCHDOG_MS = 20_000;
export const IDLE_EVICT_GRACE_MS = 60_000;
export const COMPLETE_LINGER_MS = 6_000;
export const CANCELLED_LINGER_MS = 6_000;
export const ERROR_LINGER_MS = 12_000;
export const INVENTORY_BUMP_DEBOUNCE_MS = 250;
export const TRANSPORT_STATUS_RETRY_DELAY_MS = 300;

export const ACTIVE_STATES: ReadonlySet<DownloadJobState> = new Set([
  "running",
  "cancelling",
]);

export const TERMINAL_DISPLAY_STATES: ReadonlySet<DownloadJobState> = new Set([
  "complete",
  "error",
  "cancelled",
]);
