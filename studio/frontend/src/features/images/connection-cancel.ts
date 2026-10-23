// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

/** Stop this account's runs on a saved image connection. Kept out of api.ts, which is in the startup
 *  bundle; same no-retry, abortable contract as cancelDiffusionGeneration. */
export async function cancelConnectionImageGeneration(
  signal?: AbortSignal,
): Promise<{ cancelled: boolean }> {
  const response = await authFetch(
    "/api/inference/images/generate/cancel?target=connection",
    { method: "POST", signal },
    { retryNetworkErrors: false },
  );
  if (!response.ok) throw new Error(await readFastApiError(response));
  return (await response.json()) as { cancelled: boolean };
}
