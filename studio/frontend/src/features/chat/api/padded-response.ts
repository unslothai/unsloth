// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * `/api/inference/load` and `/unload` pad their body so a proxy cannot time the
 * request out, which commits the 200 before the work finishes (see
 * `_tunnel_safe_json` in studio/backend/routes/inference.py). A proxy that then
 * gives up mid-pad leaves the client a 200 with an empty or truncated body, so a
 * client that accepts it reports an unfinished load or unload as done. These are
 * the only two routes that can do that, so they are the only ones that require a
 * payload; every other endpoint writes its body in one shot and some send none.
 *
 * Mirrored by `require_completed_padded_body` in unsloth_cli/_inference.py.
 */
export function assertCompletedPaddedBody(body: unknown, label: string): void {
  const complete =
    typeof body === "object" &&
    body !== null &&
    !Array.isArray(body) &&
    Object.keys(body).length > 0;
  if (complete) {
    return;
  }
  throw new Error(
    `${label} did not report completion: the connection closed before the server's reply arrived. Check the model's status before retrying.`,
  );
}
