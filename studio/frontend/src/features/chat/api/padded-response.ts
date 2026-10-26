// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** These routes pad their body so a proxy cannot time out; a proxy giving up leaves an empty or
 *  truncated 200. Mirrors `require_completed_padded_body` in unsloth_cli/_inference.py. */
export function assertCompletedPaddedBody(body: unknown, label: string): void {
  const complete =
    typeof body === "object" &&
    body !== null &&
    !Array.isArray(body) &&
    Object.keys(body).length > 0;
  if (complete) {
    return;
  }
  // Tagged like a failed fetch: the connection closed, so the backend's outcome is unknown.
  throw Object.assign(
    new Error(
      `${label} did not report completion: the connection closed before the server's reply arrived. Check the model's status before retrying.`,
    ),
    { unslothTransportFailure: true },
  );
}
