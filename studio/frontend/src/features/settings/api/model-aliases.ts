// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type ModelAlias = { name: string; model: string };

function parseAliases(body: unknown): ModelAlias[] {
  const aliases = (body as { aliases?: unknown } | null)?.aliases;
  if (!Array.isArray(aliases)) {
    return [];
  }
  return aliases.flatMap((entry) =>
    typeof entry?.name === "string" && typeof entry?.model === "string"
      ? [{ name: entry.name, model: entry.model }]
      : [],
  );
}

export async function loadModelAliases(): Promise<ModelAlias[]> {
  const res = await authFetch("/api/settings/model-aliases");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load model aliases"),
    );
  }
  return parseAliases(await res.json());
}

/** Replaces the whole list; the server answers with what it stored. */
export async function saveModelAliases(
  aliases: ModelAlias[],
): Promise<ModelAlias[]> {
  const res = await authFetch("/api/settings/model-aliases", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ aliases }),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to save model aliases"),
    );
  }
  return parseAliases(await res.json());
}
