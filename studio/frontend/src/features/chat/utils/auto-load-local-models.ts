// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LocalModelInfo } from "../api/chat-api";

const GGUF_REPO_SUFFIX_RE = /-GGUF(?:$|-)/i;

/** GGUF detection for a local model by backend format hint, name, or file path. */
export function localModelIsGguf(model: LocalModelInfo): boolean {
  return (
    model.model_format === "gguf" ||
    GGUF_REPO_SUFFIX_RE.test(model.id) ||
    GGUF_REPO_SUFFIX_RE.test(model.display_name) ||
    model.path.toLowerCase().endsWith(".gguf")
  );
}

export function isDirectGgufPath(path: string): boolean {
  return path.toLowerCase().endsWith(".gguf");
}

/** Local models outside the HF cache that auto-load should consider. */
export function isAutoLoadLocalModel(model: LocalModelInfo): boolean {
  return (
    model.source === "custom" ||
    model.source === "models_dir" ||
    model.source === "lmstudio"
  );
}

export function findLocalModel(
  models: LocalModelInfo[],
  id: string,
): LocalModelInfo | undefined {
  const normalized = id.trim().toLowerCase();
  if (!normalized) {
    return undefined;
  }
  return models.find((model) => {
    if (model.id.toLowerCase() === normalized) {
      return true;
    }
    if (model.path.toLowerCase() === normalized) {
      return true;
    }
    const modelId = model.model_id?.trim().toLowerCase();
    return Boolean(modelId && modelId === normalized);
  });
}

/** Prefer recently touched local models, then stable name order. */
export function sortLocalModelsForAutoLoad(
  models: LocalModelInfo[],
): LocalModelInfo[] {
  const name = (model: LocalModelInfo) =>
    model.model_id ?? model.display_name ?? model.id;
  return [...models].sort((a, b) => {
    const updatedDiff = (b.updated_at ?? 0) - (a.updated_at ?? 0);
    if (updatedDiff !== 0) {
      return updatedDiff;
    }
    return name(a).localeCompare(name(b));
  });
}
