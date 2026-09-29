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

/** Pin a cached row to its validated snapshot; `displayRepoId` stays the logical id for planning. */
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

/** On-device even for cached Hub rows, whose source stays non-`local` for companion planning. */
export function diffusionPipelineTargetIsOnDevice(target: {
  repoId: string;
  displayRepoId: string;
  source: DiffusionPickSource;
}): boolean {
  return target.source === "local" || target.repoId !== target.displayRepoId;
}

/** Pinned picks drop selected-model entries (mutable Hub revision); only companions are staged. */
export function diffusionPipelineStagingEntries<
  T extends { checkpoint?: boolean },
>(pinnedRepoId: string, planRepoId: string, entries: readonly T[]): T[] {
  return pinnedRepoId === planRepoId
    ? [...entries]
    : entries.filter((entry) => entry.checkpoint !== true);
}
