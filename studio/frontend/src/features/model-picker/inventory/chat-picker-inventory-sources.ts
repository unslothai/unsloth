// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LOCAL_MODEL_SOURCE } from "../../hub/inventory/constants.ts";

// Minimal structural shapes so these helpers stay pure and unit-testable without
// pulling in the full hub inventory row types (CachedInventoryRow / LocalInventoryRow
// are structurally assignable to these).
export interface CachedKeyRow {
  repoId: string;
  modelFormat: string;
  partial?: boolean;
  liveDownload?: boolean;
}

export interface LocalKeyRow {
  source: string;
  repoId: string | null;
  modelFormat: string;
}

interface SearchableLocalModel {
  id: string;
  display_name: string;
  model_id?: string | null;
}

function normalizeForSearch(value: string): string {
  return value.toLowerCase().replace(/[\s_.-]/g, "");
}

// A cached model and an hf_cache local row describe the same weights when repo id
// and format match; identity is compared case- and whitespace-insensitively.
function inventoryKey(repoId: string, modelFormat: string): string {
  return `${repoId.trim().toLowerCase()}\n${modelFormat.trim().toLowerCase()}`;
}

// Keys of cached models that are fully materialized. A partial or still-downloading
// row is NOT a duplicate — the hf_cache copy is the only loadable one until it finishes.
export function completeCachedModelKeys(
  cachedRows: readonly CachedKeyRow[],
): ReadonlySet<string> {
  return new Set(
    cachedRows
      .filter((row) => !row.partial && !row.liveDownload)
      .map((row) => inventoryKey(row.repoId, row.modelFormat)),
  );
}

// An hf_cache row is a duplicate only when the same model is already surfaced as a
// complete cached entry; dropping it stops the picker from listing the model twice.
export function isHfCacheDuplicate(
  row: LocalKeyRow,
  completeCachedKeys: ReadonlySet<string>,
): boolean {
  return (
    row.source === LOCAL_MODEL_SOURCE.HF_CACHE &&
    row.repoId !== null &&
    completeCachedKeys.has(inventoryKey(row.repoId, row.modelFormat))
  );
}

export function pickerLocalModelMatchesQuery(
  model: SearchableLocalModel,
  query: string,
): boolean {
  const normalizedQuery = normalizeForSearch(query.trim());
  if (!normalizedQuery) return true;
  return normalizeForSearch(
    `${model.model_id ?? ""} ${model.display_name} ${model.id}`,
  ).includes(normalizedQuery);
}
