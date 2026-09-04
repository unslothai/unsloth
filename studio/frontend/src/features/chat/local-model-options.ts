// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LocalModelInfo } from "@/features/hub";
import type { LoraModelOption } from "@/features/model-picker";

/** The device-inventory sources Chat lists as local models, deliberately the same set as the
 *  picker's `PICKER_LOCAL_SOURCES`. `/api/hub/local` scans read-only, so an Ollama row's id is an
 *  opaque `ollama-manifest:` reference; POST /load resolves it through
 *  `materialize_ollama_model_ref`, which creates the `.gguf` link on demand. */
const CHAT_LOCAL_SOURCES: ReadonlySet<LocalModelInfo["source"]> = new Set([
  "lmstudio",
  "models_dir",
  "ollama",
  "custom",
]);

function baseModelLabel(source: LocalModelInfo["source"]): string {
  switch (source) {
    case "lmstudio":
      return "LM Studio";
    case "ollama":
      return "Ollama";
    case "custom":
      return "Custom Folders";
    default:
      return "Local models";
  }
}

/** Chat's local model options, one per load id. The shared inventory keys a row on (format, path),
 *  so a directory holding both GGUF and safetensors weights arrives as two rows with distinct
 *  `inventory_id` but the SAME `id`. The selector keys on `id`, so both rows would collide on a
 *  React key and render as selected together. The compat endpoint returned one row per directory,
 *  so collapsing on `id` keeps that behaviour. */
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
    const isDirectGguf =
      model.source === "ollama" || model.path.toLowerCase().endsWith(".gguf");
    options.push({
      id: model.id,
      name: model.display_name,
      baseModel: baseModelLabel(model.source),
      updatedAt: model.updated_at ?? undefined,
      source: "local" as const,
      isGguf: isDirectGguf ? true : undefined,
      isDirectGguf: isDirectGguf ? true : undefined,
    });
  }
  return options;
}
