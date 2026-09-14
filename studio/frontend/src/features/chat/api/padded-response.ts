// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** `/api/inference/load` and `/unload` pad their body so a proxy cannot time the request out,
 *  committing the 200 before the work finishes (`_tunnel_safe_json` in routes/inference.py). A
 *  proxy giving up mid-pad leaves a 200 with an empty or truncated body, and accepting that
 *  reports an unfinished load as done. Only these two routes commit that early, so only they
 *  require a payload. Mirrored by `require_completed_padded_body` in unsloth_cli/_inference.py. */
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
