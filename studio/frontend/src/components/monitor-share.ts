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
  measured?: boolean;
}

export interface ModelBreakdown {
  models: ModelShare[];
  /** Null when RSS cannot tell the app apart from the models. */
  app: number | null;
}

const MIB_PER_GIB = 1024;

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

/** llama.cpp's inference GPUs share no backend with the displayed ones ("cuda+vulkan" includes "cuda"). */
export function inferenceGpuDisjoint(
  systemBackend: string | null | undefined,
  inferenceBackend: string | null | undefined,
): boolean {
  if (!systemBackend || !inferenceBackend) return false;
  return !inferenceBackend.split("+").includes(systemBackend);
}

/** `gpuOnly`: only models known to sit on the displayed GPUs. */
export function modelShares(
  entries: readonly {
    id: string;
    name: string;
    label?: string;
    source?: string;
    residentMib?: number;
    residentOnGpu?: boolean;
  }[],
  gpuOnly = false,
  inferenceGpuElsewhere = false,
): ModelShare[] {
  return entries
    .filter(
      (entry) =>
        (entry.residentMib ?? 0) > 0 &&
        (!gpuOnly ||
          (entry.residentOnGpu === true &&
            !(inferenceGpuElsewhere && entry.source === "chat"))),
    )
    .map((entry) => ({
      id: entry.id,
      name: entry.label || entry.name.split("/").pop() || entry.name,
      gb: (entry.residentMib ?? 0) / MIB_PER_GIB,
      ...(entry.source === "chat" ? { measured: true } : {}),
    }))
    .sort((a, b) => b.gb - a.gb);
}

/** Metal buffers can miss RSS on Apple, so the models' own size is the floor. */
export function unslothPoolGb(
  processGb: number | null,
  modelsGb: number,
  modelsInPool: boolean,
): number | null {
  if (processGb === null || !Number.isFinite(processGb)) return null;
  return modelsInPool ? Math.max(processGb, modelsGb) : processGb;
}

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
  return { models, app: app >= 0.01 ? app : null };
}

export interface MemorySegment {
  id: string;
  /** "unattributed": in use, owner unknown (VRAM, where only model weights are measured). */
  kind: "model" | "app" | "unsloth" | "other" | "unattributed" | "free";
  name?: string;
  gb: number;
}

export function memorySegments(
  share: MemoryShare,
  breakdown: ModelBreakdown | null,
  /** False on VRAM: the rest may be Unsloth's own KV cache or a training run, not other apps. */
  restIsOtherApps = true,
): MemorySegment[] {
  const parts = breakdown && breakdown.models.length > 0
    ? [
        ...breakdown.models.map((model) => ({
          id: `model:${model.id}`,
          kind: "model" as const,
          name: model.name,
          gb: model.gb,
          measured: model.measured === true,
        })),
        ...(breakdown.app !== null
          ? [{ id: "app", kind: "app" as const, gb: breakdown.app, measured: false }]
          : []),
      ]
    : null;
  // Status can be fresher than the memory sample: shrink estimates before measured rows to fit.
  const sum = (list: { gb: number }[]) => list.reduce((total, part) => total + part.gb, 0);
  const exactGb = sum(parts?.filter((part) => part.measured) ?? []);
  const restGb = sum(parts?.filter((part) => !part.measured) ?? []);
  const fit = (gb: number, room: number) => (gb > room && gb > 0 ? Math.max(0, room) / gb : 1);
  const exactScale = fit(exactGb, share.unsloth);
  const restScale = fit(restGb, share.unsloth - exactGb * exactScale);
  const unsloth: MemorySegment[] = parts
    ? parts.map(({ measured, ...part }) => ({
        ...part,
        gb: part.gb * (measured ? exactScale : restScale),
      }))
    : [{ id: "unsloth", kind: "unsloth", gb: share.unsloth }];
  return [
    ...unsloth,
    { id: "other", kind: restIsOtherApps ? "other" : "unattributed", gb: share.other },
    { id: "free", kind: "free", gb: share.free },
  ];
}

export interface PoolSplitInput {
  entries: Parameters<typeof modelShares>[0];
  showShare: boolean;
  showBreakdown: boolean;
  modelsInVram: boolean;
  inferenceGpuElsewhere: boolean;
  processGb: number | null;
  ramUsedGb: number;
  ramTotalGb: number;
  vramUsedGb: number | null;
  vramTotalGb: number;
}

export interface PoolSplit {
  ram: MemorySegment[] | null;
  vram: MemorySegment[] | null;
  noModels: boolean;
}

export function poolSplit(input: PoolSplitInput): PoolSplit {
  const { showShare, showBreakdown, modelsInVram } = input;
  if (!showShare && !showBreakdown) return { ram: null, vram: null, noModels: false };
  const sum = (list: ModelShare[]) => list.reduce((total, model) => total + model.gb, 0);
  // On a GPU host each model goes to the pool it sits in; a proven CPU model stays in RAM.
  const vramModels = modelsInVram
    ? modelShares(input.entries, true, input.inferenceGpuElsewhere)
    : [];
  const ramModels = modelShares(
    modelsInVram ? input.entries.filter((entry) => entry.residentOnGpu === false) : input.entries,
  );
  const unslothRamGb = unslothPoolGb(input.processGb, sum(ramModels), true);
  const ramBreakdown =
    showBreakdown && (!modelsInVram || ramModels.length > 0)
      ? modelBreakdown(unslothRamGb, ramModels, true)
      : null;
  const ramShare =
    showShare || ramBreakdown
      ? memoryShare(input.ramUsedGb, input.ramTotalGb, unslothRamGb)
      : null;
  // An unsized or unknown-pool model would read as Unsloth holding 0 GiB of VRAM.
  const unsized = input.entries.some(
    (entry) =>
      !((entry.residentMib ?? 0) > 0) ||
      (typeof entry.residentOnGpu !== "boolean" &&
        !(input.inferenceGpuElsewhere && entry.source === "chat")),
  );
  const vramShare =
    modelsInVram && !unsized && input.vramUsedGb !== null
      ? memoryShare(input.vramUsedGb, input.vramTotalGb, sum(vramModels))
      : null;
  return {
    ram: ramShare && memorySegments(ramShare, ramBreakdown),
    vram:
      vramShare &&
      memorySegments(vramShare, showBreakdown ? { models: vramModels, app: null } : null, false),
    noModels: showBreakdown && input.entries.length === 0,
  };
}
