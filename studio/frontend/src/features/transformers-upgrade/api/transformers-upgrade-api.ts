// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

interface InstallLatestTransformersResponse {
  success: boolean;
  version: string;
  message: string;
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
