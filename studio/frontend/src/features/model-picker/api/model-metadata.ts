// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return response.json();
}

export async function fetchModelMaxPositionEmbeddings(
  modelName: string,
  hfToken?: string | null,
  signal?: AbortSignal,
): Promise<number | null> {
  const query = hfToken?.trim()
    ? `?hf_token=${encodeURIComponent(hfToken.trim())}`
    : "";
  const response = await authFetch(
    `/api/models/config/${encodeURIComponent(modelName)}${query}`,
    { signal },
  );
  const data = await parseJsonOrThrow<{ max_position_embeddings?: unknown }>(
    response,
  );
  const value = data.max_position_embeddings;
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? Math.floor(value)
    : null;
}
