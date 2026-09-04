// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type OpenAIModel = {
  id: string;
  // Resident in memory now; the rest are downloaded and servable.
  loaded?: boolean;
  // On-disk GGUF quant. Ids stay bare for OpenAI compat, so append `:quant` to pin it.
  quant?: string;
  // Every on-disk quant of the repo, `quant` first.
  quants?: string[];
};

type ApiOpenAIModelList = {
  data?: {
    id?: unknown;
    loaded?: unknown;
    quant?: unknown;
    quants?: unknown;
    task?: unknown;
  }[];
};

const CHAT_TASKS = new Set(["text-generation"]);

export function chatModelsFromCatalog(body: unknown): OpenAIModel[] {
  const data = (body as ApiOpenAIModelList | null)?.data;
  if (!Array.isArray(data)) {
    return [];
  }
  return data.flatMap((entry) =>
    typeof entry?.id === "string" &&
    entry.id &&
    (typeof entry.task !== "string" || CHAT_TASKS.has(entry.task))
      ? [
          {
            id: entry.id,
            loaded: entry.loaded === true,
            quant: typeof entry.quant === "string" ? entry.quant : undefined,
            quants: Array.isArray(entry.quants)
              ? entry.quants.filter((q): q is string => typeof q === "string")
              : undefined,
          },
        ]
      : [],
  );
}
