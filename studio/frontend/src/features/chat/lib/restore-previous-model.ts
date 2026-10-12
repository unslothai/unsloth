// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Whether a model switch that failed after unloading the previous model may load that model back.

/**
 * A load that never got an answer did not fail on the backend: the connection closed first.
 * `/api/inference/load` keeps running when its client goes away (see `_tunnel_safe_json` in
 * routes/inference.py), so the model that was asked for is still loading or already resident.
 * Loading the previous model back from here races it and usually wins, since the rollback is
 * sent with `force_cancel_active` and queues right behind it.
 *
 * The common way to get here is reloading the page while a load is in flight: both Chromium and
 * Firefox reject the old document's pending fetch and still let its catch send a request. The
 * rollback then replaced the model the user had just loaded with the one it replaced, at that
 * model's old context, and the reloaded page read the remembered Context Length as no longer
 * applying to what was running.
 *
 * `unslothTransportFailure` is set by `authFetch` when fetch itself rejects, and by
 * `assertCompletedPaddedBody` when a padded reply is cut off. An HTTP error or a deferred error in
 * the body is an answer, and keeps the rollback.
 */
export function loadOutcomeUnknown(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    (error as { unslothTransportFailure?: unknown }).unslothTransportFailure ===
      true
  );
}

/** True when the backend reported the failure, so restoring the unloaded model cannot undo a load. */
export function shouldRestorePreviousModel(error: unknown): boolean {
  return !loadOutcomeUnknown(error);
}
