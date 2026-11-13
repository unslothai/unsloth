// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Capability filter for the On Device list. Takes Discover's capability picks and
// shares its detection, so both filter panels behave the same.

import type {
  CachedInventoryRow,
  LocalInventoryRow,
} from "@/features/hub/inventory/types";
import {
  type HubCapabilityFilter,
  detectCapabilities,
  isVideoFamilyName,
} from "./model-capabilities";
import { matchesHubCapability } from "./view-models";

function rowName(row: CachedInventoryRow | LocalInventoryRow): string {
  return row.kind === "local"
    ? `${row.id} ${row.repoId ?? ""} ${row.title} ${row.modelId ?? ""}`
    : `${row.id} ${row.repoId}`;
}

/** No picks means every capability. */
export function matchesInventoryCapabilities(
  row: CachedInventoryRow | LocalInventoryRow,
  picks: readonly HubCapabilityFilter[],
): boolean {
  if (picks.length === 0) {
    return true;
  }
  // Honor the row's own vision flag before falling back to tag detection.
  if (picks.includes("vision") && row.capabilities.supportsVision) {
    return true;
  }
  const tags = row.tags ?? undefined;
  const id = rowName(row);
  // A diffusion GGUF this host cannot build still makes images or video.
  const task =
    row.task === "image-diffusion-unsupported"
      ? isVideoFamilyName(id)
        ? "text-to-video"
        : "text-to-image"
      : row.task;
  const pipelineTag = row.pipelineTag ?? task ?? undefined;
  const view = {
    capabilities: detectCapabilities(tags, pipelineTag, id),
    result: { tags, pipelineTag, id },
  };
  return picks.some(
    (key) =>
      !(key === "text" && row.capabilities.supportsVision) &&
      matchesHubCapability(view, key),
  );
}
