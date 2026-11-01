// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

const PREFERENCE_URL = "/api/settings/model-gguf-preference";

/** Read the signed-in account's API default for a model; null means normal selection. */
export async function readGgufPreference(
  modelId: string,
  signal?: AbortSignal,
): Promise<string | null> {
  const query = new URLSearchParams({ model_id: modelId });
  const response = await authFetch(`${PREFERENCE_URL}?${query}`, { signal });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Failed to read API default quant"),
    );
  }
  return ((await response.json()) as { gguf_variant: string | null })
    .gguf_variant;
}

/** Save a quant or clear it with null, returning the value confirmed by the backend. */
export async function writeGgufPreference(
  modelId: string,
  variant: string | null,
): Promise<string | null> {
  const response = await authFetch(PREFERENCE_URL, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, gguf_variant: variant }),
  });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Failed to save API default quant"),
    );
  }
  return ((await response.json()) as { gguf_variant: string | null })
    .gguf_variant;
}
