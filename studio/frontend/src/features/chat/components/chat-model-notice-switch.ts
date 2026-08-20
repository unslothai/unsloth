// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  LoraModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker/components/model-selector/types";

/**
 * The picker metadata a "Switch back" has to carry, or undefined when the id
 * needs none.
 *
 * `handleCheckpointChange` is the picker's handler, and the picker never calls
 * it with the id alone. For a Hub row or an external model the id is enough:
 * `selectModel` finds the row in `/api/models/list` (which carries `isGguf`) or
 * `isExternalModelId` routes it. A local / fine-tuned row is in neither, so
 * with no metadata `isGguf` resolves false and the /load request drops
 * `n_parallel`, `n_batch`, `n_ubatch` and `llama_extra_args` and sizes the
 * context down the transformers path -- a different load from the one the same
 * model gets when picked from the menu.
 *
 * Mirrors what the picker itself supplies for these rows, field for field:
 * `localDirectGgufMeta` for a single .gguf file, and the fine-tuned list's own
 * `selectionMeta` for everything else.
 */
export function chatModelSwitchMeta(
  modelId: string,
  loraModels: readonly LoraModelOption[],
): ModelSelectorChangeMeta | undefined {
  const row = loraModels.find((model) => model.id === modelId);
  if (!row) return undefined;
  const isLocal = row.source === "local";
  return {
    source: isLocal ? "local" : row.source === "exported" ? "exported" : "lora",
    // A local folder is not an adapter, and neither is a merged or GGUF export.
    isLora:
      !isLocal && row.exportType !== "merged" && row.exportType !== "gguf",
    // Already on disk: without this the load shows a download progress bar for
    // a model that has nothing to download.
    isDownloaded: true,
    // Set only by chatLocalModelOptions, and only for a single .gguf file.
    isGguf: row.isGguf === true,
  };
}
