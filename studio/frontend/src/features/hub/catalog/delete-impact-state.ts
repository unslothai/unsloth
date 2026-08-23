// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DeleteImpact } from "../inventory";

/**
 * The status the delete endpoint gives a holder that releases itself -- a load, a download, or
 * another delete -- as opposed to a 400 the user has to clear (unload the model, delete the
 * models that share its assets).
 */
const DELETE_BLOCK_RETRY_LATER = 409;
/** A block the backend could not substantiate: it says "I don't know", not "something holds this". */
const LOAD_STATE_UNVERIFIABLE = 503;

export const DELETE_BLOCK_REPOLL_MS = 2_000;
export const DELETE_BLOCK_REPOLL_CAP_MS = 30_000;

export function isUnverifiable(impact: DeleteImpact | null): boolean {
  return impact?.delete_block?.status_code === LOAD_STATE_UNVERIFIABLE;
}

/**
 * Whether the preview positively knows something holds this delete.
 *
 * Only that: the delete endpoint re-runs every guard and refuses authoritatively, so a preview
 * that has not landed or could not be reached leaves Delete enabled rather than parking a
 * destructive action behind a cache scan. A 503 block is the same answer from the other side --
 * the backend saying it cannot read its own load state -- so it warns and leaves Delete enabled
 * too, rather than greying the button out behind a poll that has nothing to wait for.
 */
export function isDeleteBlocked(impact: DeleteImpact | null): boolean {
  if (impact === null) return false;
  return (
    (Boolean(impact.delete_block) && !isUnverifiable(impact)) ||
    impact.blocked_by.length > 0
  );
}

export function shouldRefreshDeleteImpactOnWake(impact: DeleteImpact | null): boolean {
  return impact === null || isDeleteBlocked(impact);
}

/**
 * How long to wait before reading the preview again, or null to stop polling.
 *
 * Only a retry-later block is worth watching on a timer: Delete recovers on its own when the
 * download or load finishes. User-cleared blocks, shared-asset blocks and unavailable previews
 * are re-read when the dialog returns to the foreground, without repeatedly walking the whole
 * HF cache. The delay doubles up to the cap because a download the dialog is waiting on can run
 * for many minutes.
 */
export function repollDelayMs(
  impact: DeleteImpact | null,
  previous: number | null,
): number | null {
  if (
    impact?.delete_block?.status_code !== DELETE_BLOCK_RETRY_LATER ||
    impact.delete_block.retryable === false
  ) {
    return null;
  }
  if (previous === null) {
    return DELETE_BLOCK_REPOLL_MS;
  }
  return Math.min(previous * 2, DELETE_BLOCK_REPOLL_CAP_MS);
}
