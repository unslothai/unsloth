// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settlement ownership is shared, not per call, or a mid-wait refresh would end the wait early.

import {
  disposableTimeoutSignal,
  pollSignal,
  type PollSignal,
} from "@/features/hub/lib/abort-signals";

let outstanding = 0;

export function serverModelWaitOutstanding(): boolean {
  return outstanding > 0;
}

/** Also released when `signal` aborts, since not every request here is cancellable. */
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

/** fetch has no timeout; this caps a half-open poll (~200x a healthy read). */
export const STATUS_POLL_TIMEOUT_MS = 30_000;

/** Callers MUST dispose the signal once the read settles. */
export function statusPollSignal(parent?: AbortSignal): PollSignal {
  return parent
    ? pollSignal(parent, STATUS_POLL_TIMEOUT_MS)
    : disposableTimeoutSignal(STATUS_POLL_TIMEOUT_MS);
}
