// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeConfig } from "../../types";
import { syncSubcategoryConfigsForCategoryUpdate } from "./edge-sync";
import { applyRenameToConfigs } from "./reference-sync";

/** What a block edit (`current` -> `next`, already in `configs`) changes in the other blocks:
 * subcategories of a category, references to a renamed block, models of a provider that
 * toggled local. Shared by Recipe Studio's store and Flows' recipe forms. */
export function syncConfigsForPatch(
  configs: Record<string, NodeConfig>,
  current: NodeConfig,
  next: NodeConfig,
): Record<string, NodeConfig> {
  const oldName = current.name;
  const newName = next.name;
  const nameChanged = oldName !== newName;
  let result = syncSubcategoryConfigsForCategoryUpdate(
    current,
    next,
    configs,
    oldName,
    newName,
    nameChanged,
  );
  if (nameChanged) {
    result = applyRenameToConfigs(result, oldName, newName);
  }
  // Keep linked model_config nodes in sync when a provider toggles local/external.
  if (
    current.kind === "model_provider" &&
    next.kind === "model_provider" &&
    (current.is_local === true) !== (next.is_local === true)
  ) {
    for (const [cfgId, cfg] of Object.entries(result)) {
      if (cfg.kind !== "model_config" || cfg.provider !== next.name) {
        continue;
      }
      result = {
        ...result,
        [cfgId]: {
          ...cfg,
          model: "",
          // biome-ignore lint/style/useNamingConvention: api schema
          gguf_variant: undefined,
        },
      };
    }
  }
  return result;
}
