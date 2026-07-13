// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

interface InstallLatestTransformersResponse {
  success: boolean;
  version: string;
  message: string;
  /** The server unloaded the active chat model before the swap (set even on a
   *  structured failure, so callers can restore their model state). */
  model_unloaded?: boolean;
  /** On a version-mismatch failure: the release that superseded the requested
   *  one, so Retry can use it. */
  latest_version?: string | null;
}

/** Consented install of the latest transformers into the sidecar; synchronous, can take minutes. */
export async function installLatestTransformers(
  version: string,
): Promise<InstallLatestTransformersResponse> {
  const response = await authFetch("/api/inference/install-latest-transformers", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ version }),
  });
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as InstallLatestTransformersResponse;
}
