// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Dismissal lives on the backend, not in localStorage: an Unsloth origin moves
// whenever port 8888 is taken, and a per-origin store would hand out a fresh
// notice on every new origin -- the same defect the Xet notice already hit.

import { authFetch } from "@/features/auth";

/** Stop offering the advice at this allocation.
 *
 * `current_gb` is what the user is dismissing AT, so raising the allocation and
 * running short again can say so once more.
 *
 * Never throws and never retries: the caller has already closed the dialog, so a
 * failure here costs at most one repeat of the notice on a later load, whereas a
 * retry whose predecessor reached the backend is a second pointless write.
 */
export async function dismissCarveoutNotice(currentGb: number | null): Promise<boolean> {
  try {
    const response = await authFetch(
      "/api/settings/igpu-carveout-notice/dismiss",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ current_gb: currentGb }),
      },
      { retryNetworkErrors: false },
    );
    return response.ok;
  } catch {
    return false;
  }
}
