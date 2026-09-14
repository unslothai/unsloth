// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Shared by the two loops that wait for the server to settle: the mount observer and the
// send-path probe. Both poll on the same selection every refresh writes, so ownership of
// settlement cannot be a per-call argument -- a refresh landing mid-wait would publish the
// outgoing model as the pick and end the wait on it.

import {
  disposableTimeoutSignal,
  pollSignal,
  type PollSignal,
} from "@/features/hub/lib/abort-signals";

let outstanding = 0;

/** Whether some loop is waiting for the server to settle and will publish the answer. */
export function serverModelWaitOutstanding(): boolean {
  return outstanding > 0;
}

/**
 * Register a wait. Released by the returned call and by ``signal`` aborting, since the
 * requests these loops make are not all cancellable and one of them stalling would keep
 * the gate up past the page that raised it.
 */
export function beginServerModelWait(signal?: AbortSignal): () => void {
  outstanding += 1;
  let held = true;
  const release = () => {
    if (!held) return;
    held = false;
    outstanding -= 1;
    signal?.removeEventListener("abort", release);
  };
  signal?.addEventListener("abort", release, { once: true });
  return release;
}

/**
 * Per-request cap for a status poll, so the loop's own deadline is real: fetch has no
 * timeout of its own, and a half-open connection would otherwise park a poll forever and
 * hold both the gate above and the send's model-loading lease. Two orders of magnitude
 * over a healthy read (~140ms against an idle backend), so only a wedged one hits it.
 */
export const STATUS_POLL_TIMEOUT_MS = 30_000;

/** The capped signal for one status read. Callers MUST dispose it once the read settles. */
export function statusPollSignal(parent?: AbortSignal): PollSignal {
  return parent
    ? pollSignal(parent, STATUS_POLL_TIMEOUT_MS)
    : disposableTimeoutSignal(STATUS_POLL_TIMEOUT_MS);
}
