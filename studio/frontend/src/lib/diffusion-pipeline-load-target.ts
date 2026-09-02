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

/** Whether the eventual load target is already on this host.
 *
 * A cached Hub row deliberately retains Hub provenance for companion planning, so its source is
 * not `local`; the pinned physical identity is what makes the eventual pipeline load on-device.
 */
export function diffusionPipelineTargetIsOnDevice(target: {
  repoId: string;
  displayRepoId: string;
  source: DiffusionPickSource;
}): boolean {
  return target.source === "local" || target.repoId !== target.displayRepoId;
}

/** Keep selected-model staging from replacing an inspected immutable snapshot.
 *
 * Planning uses the logical Hub id so it can discover external pre-quantized components and
 * companion repos. When the pick is pinned, however, selected-model entries describe a mutable
 * Hub revision rather than the snapshot whose manifest was validated. Drop those entries and
 * stage only external companions; the eventual load remains pinned to `pinnedRepoId`.
 */
export function diffusionPipelineStagingEntries<
  T extends { checkpoint?: boolean },
>(pinnedRepoId: string, planRepoId: string, entries: readonly T[]): T[] {
  return pinnedRepoId === planRepoId
    ? [...entries]
    : entries.filter((entry) => entry.checkpoint !== true);
}
