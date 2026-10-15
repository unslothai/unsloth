// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LocalModelInfo } from "@/features/hub";
import type { LoraModelOption } from "@/features/model-picker";

/** Same set as `PICKER_LOCAL_SOURCES`. Ollama ids are resolved to a `.gguf` link by /load. */
const CHAT_LOCAL_SOURCES: ReadonlySet<LocalModelInfo["source"]> = new Set([
  "lmstudio",
  "omlx",
  "models_dir",
  "ollama",
  "hermes",
  "custom",
]);

function baseModelLabel(source: LocalModelInfo["source"]): string {
  switch (source) {
    case "lmstudio":
      return "LM Studio";
    case "omlx":
      return "oMLX";
    case "ollama":
      return "Ollama";
    case "hermes":
      return "Hermes";
    case "custom":
      return "Custom Folders";
    default:
      return "Local models";
  }
}

/** One option per `id`: GGUF and safetensors in one dir would otherwise collide on React keys. */
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
