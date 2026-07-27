// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type OpenAIModel = {
  id: string;
  // Resident in memory now; the rest are downloaded and servable.
  loaded?: boolean;
  // On-disk GGUF quant. Ids stay bare for OpenAI compat, so append `:quant` to pin it.
  quant?: string;
};

type ApiOpenAIModelList = {
  data?: { id?: unknown; loaded?: unknown; quant?: unknown }[];
};

/**
 * The models this server can serve: `/v1/models` returns exactly the ids
 * `/v1/chat/completions` resolves against, and accepts the UI session JWT.
 */
export async function listOpenAIModels(): Promise<OpenAIModel[]> {
  const res = await authFetch("/v1/models");
  if (!res.ok) {
    throw new Error(`Failed to list models (${res.status})`);
  }
  const body = (await res.json()) as ApiOpenAIModelList;
  if (!Array.isArray(body?.data)) {
    return [];
  }
  return body.data.flatMap((entry) =>
    typeof entry?.id === "string" && entry.id
      ? [
          {
            id: entry.id,
            loaded: entry.loaded === true,
            quant: typeof entry.quant === "string" ? entry.quant : undefined,
          },
        ]
      : [],
  );
}
