// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

/** GGUF launch settings an alias layers over its model's saved settings; omitted fields keep them. */
export type ModelAliasLoadConfig = {
  max_seq_length?: number;
  kv_cache_dtype?: string;
  speculative_type?: string;
  n_parallel?: number;
  gpu_layers?: number | "auto";
  n_cpu_moe?: number;
  disable_vision?: boolean;
  // Settings a newer release stored travel back untouched.
  [key: string]: unknown;
};

export type ModelAlias = {
  name: string;
  model: string;
  load_config?: ModelAliasLoadConfig | null;
  // Entries are kept whole, so a save never drops a field this page does not know.
  [key: string]: unknown;
};

export type ModelAliasList = {
  aliases: ModelAlias[];
  /** Version of the stored list; a save naming an older one is refused. */
  etag: string | null;
};

/** The list changed since this page read it (another tab, or settings this page cannot show). */
export class ModelAliasConflictError extends Error {}

function parseAliasList(body: unknown): ModelAliasList {
  const raw = body as { aliases?: unknown; etag?: unknown } | null;
  const aliases = Array.isArray(raw?.aliases) ? raw.aliases : [];
  return {
    aliases: aliases.flatMap((entry) =>
      typeof entry?.name === "string" && typeof entry?.model === "string"
        ? [{ ...entry } as ModelAlias]
        : [],
    ),
    etag: typeof raw?.etag === "string" ? raw.etag : null,
  };
}

export async function loadModelAliases(): Promise<ModelAliasList> {
  const res = await authFetch("/api/settings/model-aliases");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load model aliases"),
    );
  }
  return parseAliasList(await res.json());
}

/** Replaces the whole list read at `etag`; the server answers with what it stored. */
export async function saveModelAliases(
  aliases: ModelAlias[],
  etag: string | null,
): Promise<ModelAliasList> {
  const res = await authFetch("/api/settings/model-aliases", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ aliases, etag }),
  });
  if (!res.ok) {
    const message = await readFastApiError(res, "Failed to save model aliases");
    throw res.status === 409
      ? new ModelAliasConflictError(message)
      : new Error(message);
  }
  return parseAliasList(await res.json());
}
