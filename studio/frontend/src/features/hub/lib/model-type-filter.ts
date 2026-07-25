// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Type filter for the On Device list. Mirrors the hub Discover capability
// options and shares its detection, so both dropdowns behave the same.

import type {
  CachedInventoryRow,
  LocalInventoryRow,
} from "@/features/hub/inventory/types";
import { type CapabilityKey, detectCapabilities } from "./model-capabilities";

export type ModelTypeFilter =
  | "all"
  | "reasoning"
  | "vision"
  | "audio"
  | "embedding"
  | "diffusion";

export const MODEL_TYPE_FILTER_OPTIONS: ReadonlyArray<{
  value: ModelTypeFilter;
  label: string;
}> = [
  { value: "all", label: "All types" },
  { value: "reasoning", label: "Reasoning" },
  { value: "vision", label: "Vision" },
  { value: "audio", label: "Audio" },
  { value: "embedding", label: "Embeddings" },
  { value: "diffusion", label: "Image generation" },
];

function rowName(row: CachedInventoryRow | LocalInventoryRow): string {
  return row.kind === "local"
    ? `${row.id} ${row.repoId ?? ""} ${row.title} ${row.modelId ?? ""}`
    : `${row.id} ${row.repoId}`;
}

export function matchesModelType(
  row: CachedInventoryRow | LocalInventoryRow,
  filter: ModelTypeFilter,
): boolean {
  if (filter === "all") return true;
  // Honor the row's own vision flag before falling back to tag detection.
  if (filter === "vision" && row.capabilities.supportsVision) return true;
  const caps = detectCapabilities(
    row.tags ?? undefined,
    row.pipelineTag ?? undefined,
    rowName(row),
  );
  return caps.some((cap: { key: CapabilityKey }) => cap.key === filter);
}
