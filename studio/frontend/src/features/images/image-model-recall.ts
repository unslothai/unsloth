// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { explicitFamily } from "../model-picker/components/model-selector/family-override.ts";

export interface RememberedImageModel {
  repoId: string;
  kind: "gguf" | "single_file" | "pipeline";
  filename?: string;
  familyOverride?: string;
}

const KEY = "unsloth:images:last-model";

export function readImageModel(): RememberedImageModel | null {
  try {
    const value = JSON.parse(localStorage.getItem(KEY) ?? "null");
    if (!value || typeof value.repoId !== "string" || !value.repoId.trim())
      return null;
    if (!["gguf", "single_file", "pipeline"].includes(value.kind)) return null;
    if (
      value.kind !== "pipeline" &&
      (typeof value.filename !== "string" || !value.filename)
    )
      return null;
    const family = explicitFamily(value.familyOverride);
    return {
      repoId: value.repoId,
      kind: value.kind,
      ...(typeof value.filename === "string"
        ? { filename: value.filename }
        : {}),
      ...(family ? { familyOverride: family } : {}),
    };
  } catch {
    return null;
  }
}

export function rememberImageModel(model: RememberedImageModel): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(model));
  } catch {
    // storage can be unavailable in private browsing
  }
}

export function matchesRememberedModel(
  model: RememberedImageModel,
  status: {
    loaded: boolean;
    repo_id?: string | null;
    model_kind?: string | null;
    gguf_filename?: string | null;
  },
): boolean {
  return (
    status.loaded &&
    status.repo_id === model.repoId &&
    status.model_kind === model.kind &&
    (model.kind === "pipeline" || status.gguf_filename === model.filename)
  );
}
