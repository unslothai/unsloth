// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LocalModelInfo } from "@/features/hub/inventory/api";
import type { LoraModelOption } from "@/features/model-picker/components/model-selector/types";

/** The device-inventory sources Chat lists as local models.
 *
 * Deliberately the same set as the picker's `PICKER_LOCAL_SOURCES`, and `ollama` is
 * deliberately absent from both. The compat endpoint Chat used to call materialized a
 * `.gguf` link eagerly and handed back a loadable path, while `/api/hub/local` scans
 * read-only and returns an opaque `ollama-manifest:` reference that only
 * `materialize_ollama_model_ref` can resolve. Nothing calls that yet, so offering these
 * rows would produce entries that fail on selection rather than load.
 */
const CHAT_LOCAL_SOURCES: ReadonlySet<LocalModelInfo["source"]> = new Set([
  "lmstudio",
  "models_dir",
  "custom",
]);

function baseModelLabel(source: LocalModelInfo["source"]): string {
  switch (source) {
    case "lmstudio":
      return "LM Studio";
    case "custom":
      return "Custom Folders";
    default:
      return "Local models";
  }
}

/** Chat's local model options, one per load id.
 *
 * The shared inventory keys a row on (format, path), so a directory holding both GGUF and
 * safetensors weights arrives as two rows with distinct `inventory_id` but the SAME `id`.
 * The selector keys on `id` (`key={adapter.id}`, `selected={value === adapter.id}`), so
 * both rows would collide on a React key and render as selected together. The compat
 * endpoint returned one row per directory, so collapsing on `id` keeps that behaviour.
 */
export function chatLocalModelOptions(
  rows: readonly LocalModelInfo[],
): LoraModelOption[] {
  const options: LoraModelOption[] = [];
  const seen = new Set<string>();
  for (const model of rows) {
    if (!CHAT_LOCAL_SOURCES.has(model.source) || seen.has(model.id)) {
      continue;
    }
    seen.add(model.id);
    options.push({
      id: model.id,
      name:
        model.source === "lmstudio" && model.model_id
          ? model.model_id
          : model.display_name,
      baseModel: baseModelLabel(model.source),
      updatedAt: model.updated_at ?? undefined,
      source: "local" as const,
    });
  }
  return options;
}
