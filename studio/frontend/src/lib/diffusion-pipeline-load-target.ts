// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type DiffusionPickSource = "hub" | "lora" | "exported" | "local" | "external";

const trimmed = (value: string | null | undefined): string | null =>
  typeof value === "string" && value.trim().length > 0 ? value.trim() : null;

export function isPinnedDiffusionLoadId(
  model: string,
  loadId: string | null | undefined,
): boolean {
  const pinned = trimmed(loadId);
  return Boolean(pinned && pinned !== model.trim());
}

/** Resolve a cached pipeline row to the snapshot that established its manifest. */
export function diffusionPipelineLoadTarget(
  model: string,
  meta: { loadId?: string | null; source: DiffusionPickSource },
): { repoId: string; displayRepoId: string; source: DiffusionPickSource } {
  const loadId = trimmed(meta.loadId);
  return isPinnedDiffusionLoadId(model, loadId)
    ? { repoId: loadId!, displayRepoId: model, source: "local" }
    : { repoId: model, displayRepoId: model, source: meta.source };
}
