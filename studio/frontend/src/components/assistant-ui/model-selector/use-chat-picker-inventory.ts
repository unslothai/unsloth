// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type CachedInventoryRow,
  type LocalInventoryRow,
  type LocalSource,
  useHubInventory,
} from "@/features/inventory";
import { useMemo } from "react";

type ChatPickerInventoryOptions = {
  enabled?: boolean;
};

export function localModelDisplayName(model: LocalInventoryRow): string {
  return model.repoId?.trim() || model.title;
}

export function localModelGroupLabel(source: LocalSource): string {
  switch (source) {
    case "hf_cache":
      return "HF Cache";
    case "lmstudio":
      return "LM Studio";
    case "ollama":
      return "Ollama";
    case "models_dir":
      return "Local models";
    case "custom":
      return "Custom folders";
    default:
      return "Local";
  }
}

function isPickerLocalModel(model: LocalInventoryRow): boolean {
  return (
    model.source === "hf_cache" ||
    model.source === "lmstudio" ||
    model.source === "ollama" ||
    model.source === "models_dir" ||
    model.source === "custom"
  );
}

function sortLocalModels(models: LocalInventoryRow[]): LocalInventoryRow[] {
  return [...models].sort((a, b) => {
    const aUnsloth = (a.repoId ?? "").startsWith("unsloth/") ? 0 : 1;
    const bUnsloth = (b.repoId ?? "").startsWith("unsloth/") ? 0 : 1;
    if (aUnsloth !== bUnsloth) {
      return aUnsloth - bUnsloth;
    }
    const sourceCmp = localModelGroupLabel(a.source).localeCompare(
      localModelGroupLabel(b.source),
    );
    if (sourceCmp !== 0) {
      return sourceCmp;
    }
    return localModelDisplayName(a).localeCompare(localModelDisplayName(b));
  });
}

export function useChatPickerInventory(
  options: ChatPickerInventoryOptions = {},
): {
  cachedGguf: CachedInventoryRow[];
  cachedModels: CachedInventoryRow[];
  cachedReady: boolean;
  inventoryError: string | null;
  localModels: LocalInventoryRow[];
  refreshInventory: () => Promise<void>;
} {
  const inventory = useHubInventory({
    kind: "models",
    enabled: options.enabled,
    includeLocal: true,
  });
  const localModels = useMemo(
    () =>
      sortLocalModels(inventory.localRows.filter(isPickerLocalModel)),
    [inventory.localRows],
  );
  const cachedGguf = useMemo(
    () => inventory.cachedRows.filter((row) => row.modelFormat === "gguf"),
    [inventory.cachedRows],
  );
  const cachedModels = useMemo(
    () => inventory.cachedRows.filter((row) => row.modelFormat !== "gguf"),
    [inventory.cachedRows],
  );

  return {
    cachedGguf,
    cachedModels,
    cachedReady: inventory.downloadedReady,
    inventoryError: inventory.inventoryError
      ? "Couldn't scan on-device models"
      : null,
    localModels,
    refreshInventory: inventory.refreshInventory,
  };
}
