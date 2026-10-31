// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface MemoryShare {
  unsloth: number;
  other: number;
  free: number;
}

export interface ModelShare {
  id: string;
  name: string;
  gb: number;
}

export interface ModelBreakdown {
  models: ModelShare[];
  /** The rest of Unsloth's share: the server, the UI process, caches. Null when it cannot be told apart. */
  app: number | null;
}

const MIB_PER_GIB = 1024;

/** Splits one pool's used memory into Unsloth's and everyone else's. Unsloth never exceeds what is used. */
export function memoryShare(
  usedGb: number,
  totalGb: number,
  unslothGb: number | null,
): MemoryShare | null {
  if (!(totalGb > 0) || unslothGb === null || !Number.isFinite(unslothGb)) {
    return null;
  }
  const used = Math.min(Math.max(0, usedGb), totalGb);
  const unsloth = Math.min(Math.max(0, unslothGb), used);
  return { unsloth, other: used - unsloth, free: totalGb - used };
}

/** Loaded models that report a size, largest first. `gpuOnly`: only models known to sit on the reported GPUs (VRAM split). */
export function modelShares(
  entries: readonly {
    id: string;
    name: string;
    label?: string;
    residentMib?: number;
    residentOnGpu?: boolean;
  }[],
  gpuOnly = false,
): ModelShare[] {
  return entries
    .filter((entry) => (entry.residentMib ?? 0) > 0 && (!gpuOnly || entry.residentOnGpu === true))
    .map((entry) => ({
      id: entry.id,
      // "unsloth/Qwen3.5-0.8B-GGUF" -> "Qwen3.5-0.8B-GGUF": the owner only eats the row's width.
      name: entry.label || entry.name.split("/").pop() || entry.name,
      gb: (entry.residentMib ?? 0) / MIB_PER_GIB,
    }))
    .sort((a, b) => b.gb - a.gb);
}

/**
 * Unsloth's share of a pool. On Apple, Metal buffers are not always counted in RSS, so the process
 * tree can read below the models it holds; the models' own figure is the floor then.
 */
export function unslothPoolGb(
  processGb: number | null,
  modelsGb: number,
  modelsInPool: boolean,
): number | null {
  if (processGb === null || !Number.isFinite(processGb)) return null;
  return modelsInPool ? Math.max(processGb, modelsGb) : processGb;
}

/**
 * `modelsInPool`: the models sit inside `unslothGb` (unified memory, or a CPU host), so the app's own
 * share is what is left after them. On a discrete GPU the models are in VRAM and `unslothGb` is RAM,
 * so the app row is the whole RAM figure.
 */
export function modelBreakdown(
  unslothGb: number | null,
  models: ModelShare[],
  modelsInPool: boolean,
): ModelBreakdown {
  if (unslothGb === null || !Number.isFinite(unslothGb)) {
    return { models, app: null };
  }
  const inPool = modelsInPool ? models.reduce((sum, model) => sum + model.gb, 0) : 0;
  const app = unslothGb - inPool;
  // Under 10 MiB left means RSS could not tell the app apart from the models: no row beats "0.00 GiB".
  return { models, app: app >= 0.01 ? app : null };
}

export interface MemorySegment {
  id: string;
  /** "unattributed": in use, owner unknown (VRAM, where only model weights are measured). */
  kind: "model" | "app" | "unsloth" | "other" | "unattributed" | "free";
  /** Set for models; the rest are labelled by kind. */
  name?: string;
  gb: number;
}

/** One bar for one pool: Unsloth (or its models and the app, when broken down), other apps, free. */
export function memorySegments(
  share: MemoryShare,
  breakdown: ModelBreakdown | null,
  /** False on VRAM: the rest may be Unsloth's own KV cache or a training run, not other apps. */
  restIsOtherApps = true,
): MemorySegment[] {
  // Nothing loaded: splitting would only rename Unsloth's segment to "app".
  const unsloth: MemorySegment[] = breakdown && breakdown.models.length > 0
    ? [
        ...breakdown.models.map((model) => ({
          id: `model:${model.id}`,
          kind: "model" as const,
          name: model.name,
          gb: model.gb,
        })),
        ...(breakdown.app !== null
          ? [{ id: "app", kind: "app" as const, gb: breakdown.app }]
          : []),
      ]
    : [{ id: "unsloth", kind: "unsloth", gb: share.unsloth }];
  return [
    ...unsloth,
    { id: "other", kind: restIsOtherApps ? "other" : "unattributed", gb: share.other },
    { id: "free", kind: "free", gb: share.free },
  ];
}
