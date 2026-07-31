// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  LOCAL_MODEL_SOURCE,
  type LocalSource,
} from "../../hub/inventory/constants.ts";

export interface PickerCachedRow {
  repoId: string;
  modelFormat: string;
  partial?: boolean;
  liveDownload?: boolean;
}

export interface PickerLocalRow {
  loadId: string;
  repoId: string | null;
  title: string;
  source: LocalSource;
  modelId?: string | null;
  displayName?: string;
  path: string;
  modelFormat: string;
  capabilities: { canChat: boolean };
  updatedAt: number | null;
}

export interface PickerLocalModel {
  id: string;
  display_name: string;
  path: string;
  source: Exclude<LocalSource, "ollama">;
  model_id?: string | null;
  model_format?: string | null;
  updated_at?: number | null;
}

const PICKER_LOCAL_SOURCES: ReadonlySet<LocalSource> = new Set([
  LOCAL_MODEL_SOURCE.LMSTUDIO,
  LOCAL_MODEL_SOURCE.MODELS_DIR,
  LOCAL_MODEL_SOURCE.HF_CACHE,
  LOCAL_MODEL_SOURCE.CUSTOM,
]);

function normalizeForSearch(value: string): string {
  return value.toLowerCase().replace(/[\s_.-]/g, "");
}

function inventoryKey(repoId: string, modelFormat: string): string {
  return `${repoId.trim().toLowerCase()}\u0000${modelFormat.toLowerCase()}`;
}

function completeCachedModelKeys(
  cachedRows: readonly PickerCachedRow[],
): ReadonlySet<string> {
  return new Set(
    cachedRows
      .filter((row) => !row.partial && !row.liveDownload)
      .map((row) => inventoryKey(row.repoId, row.modelFormat)),
  );
}

export function buildPickerLocalModels(
  cachedRows: readonly PickerCachedRow[],
  localRows: readonly PickerLocalRow[],
): PickerLocalModel[] {
  const cachedModelKeys = completeCachedModelKeys(cachedRows);
  return localRows
    .filter(
      (row) =>
        PICKER_LOCAL_SOURCES.has(row.source) &&
        row.capabilities.canChat &&
        !(
          row.source === LOCAL_MODEL_SOURCE.HF_CACHE &&
          row.repoId &&
          cachedModelKeys.has(inventoryKey(row.repoId, row.modelFormat))
        ),
    )
    .map((row) => ({
      id: row.loadId,
      display_name: row.displayName ?? row.title,
      path: row.path,
      source:
        row.source === LOCAL_MODEL_SOURCE.HF_CACHE
          ? LOCAL_MODEL_SOURCE.MODELS_DIR
          : (row.source as PickerLocalModel["source"]),
      model_id: row.modelId ?? row.repoId,
      model_format: row.modelFormat,
      updated_at: row.updatedAt,
    }));
}

export function pickerLocalModelMatchesQuery(
  model: PickerLocalModel,
  query: string,
): boolean {
  const normalizedQuery = normalizeForSearch(query.trim());
  if (!normalizedQuery) return true;
  return normalizeForSearch(
    `${model.model_id ?? ""} ${model.display_name} ${model.id}`,
  ).includes(normalizedQuery);
}
