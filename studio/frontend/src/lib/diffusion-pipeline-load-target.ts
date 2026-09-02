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

/** Resolve a cached pipeline row to the snapshot that established its manifest.
 *
 * Keep the pick's provenance: a cached Hub row still needs a Hub download plan for
 * task-specific companions. `repoId` is only the eventual physical load target;
 * `displayRepoId` remains the logical identity used to plan and detect the family.
 */
export function diffusionPipelineLoadTarget(
  model: string,
  meta: { loadId?: string | null; source: DiffusionPickSource },
): { repoId: string; displayRepoId: string; source: DiffusionPickSource } {
  const loadId = trimmed(meta.loadId);
  if (loadId && isPinnedDiffusionLoadId(model, loadId)) {
    return { repoId: loadId, displayRepoId: model, source: meta.source };
  }
  return { repoId: model, displayRepoId: model, source: meta.source };
}

/** Pick the post-staging load identity without discarding a pinned manifest revision. */
export function stagedDiffusionLoadTarget(
  pinnedRepoId: string,
  planRepoId: string,
  entries: readonly { checkpoint?: boolean }[],
): string {
  return entries.some((entry) => entry.checkpoint === true)
    ? planRepoId
    : pinnedRepoId;
}
