// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * What the Advanced reseed effect does on a run. A failed load keeps the resident build (and its record), so the
 * record key does not move and the selects would keep advertising the refused pick: the page bumps a failure nonce
 * after re-reading status, and that run reseeds from what is loaded, or back to Auto when nothing is.
 *  - "record": copy the loaded build's engaged values into the selects.
 *  - "auto": a failed load left nothing loaded, so nothing engaged: show Auto rather than the refused pick.
 *  - "none": nothing to do (no record yet, or a plain unload the user asked for keeps their picks).
 */
export type ReseedAction = "record" | "auto" | "none";

export function reseedAction(
  status: { loaded?: boolean | null } | null | undefined,
  hasRecord: boolean,
  afterFailedLoad: boolean,
): ReseedAction {
  if (status?.loaded) return hasRecord ? "record" : "none";
  if (afterFailedLoad && status) return "auto";
  return "none";
}
