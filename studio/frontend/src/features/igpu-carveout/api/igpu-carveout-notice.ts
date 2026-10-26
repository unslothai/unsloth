// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Dismissal lives on the backend: the origin port changes, so localStorage would re-notice.

import { authFetch } from "@/features/auth";

/** Never throws or retries: a failure costs at most one repeat of the notice. */
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
