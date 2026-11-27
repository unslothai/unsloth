// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ExportMethod } from "../constants";

export type SourceMode = "checkpoint" | "model";

function safePathSegment(
  value: string | null | undefined,
  fallback = "model",
  maxLength = 250,
): string {
  const safe = (value ?? "")
    .replace(/[^a-zA-Z0-9._-]/g, "-")
    .replace(/^[._-]+|[._-]+$/g, "")
    .slice(0, maxLength)
    .replace(/[._-]+$/g, "");
  return safe || fallback;
}

export function buildRelativeSaveDirectory(
  exportMethod: ExportMethod | null,
  sourceMode: SourceMode,
  sourceBaseModelName: string,
  selectedModelIdx: string | null,
  checkpoint: string | null,
): string {
  const suffix =
    exportMethod === "gguf"
      ? "GGUF"
      : exportMethod === "lora"
        ? "adapter"
        : "merged";
  if (sourceMode === "checkpoint" && selectedModelIdx && checkpoint) {
    if (exportMethod !== "gguf") {
      return `${selectedModelIdx}/${checkpoint}-${suffix}`;
    }
    if (checkpoint !== selectedModelIdx) {
      return `${safePathSegment(`${selectedModelIdx}-${checkpoint}`)}-GGUF`;
    }
  }
  // Name from the model id to avoid "model/null".
  const rawName =
    sourceMode === "checkpoint"
      ? (checkpoint ?? selectedModelIdx ?? sourceBaseModelName)
      : sourceBaseModelName;
  return `${safePathSegment(rawName)}-${suffix}`;
}

export function saveDirectoryField(
  custom: string | null,
  defaultSaveDirectory: string,
): { inputValue: string; saveDirectory: string } {
  // Echoing the trimmed path back into the input would drop a space as it is
  // typed and refill a cleared field with the default.
  return {
    inputValue: custom ?? defaultSaveDirectory,
    saveDirectory: custom?.trim() || defaultSaveDirectory,
  };
}
