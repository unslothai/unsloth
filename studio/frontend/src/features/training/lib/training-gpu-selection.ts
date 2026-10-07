// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";
import type { TrainingParallelismMode } from "../types/config";

import type { SystemGpuDevice } from "@/hooks/gpu-selection";

/** Physical devices explicitly selected for the training preview. */
export function selectedTrainingPreviewDevices(
  mode: TrainingParallelismMode,
  selectedGpuIds: number[] | null,
  devices: readonly SystemGpuDevice[],
): SystemGpuDevice[] {
  if (mode === "auto" || selectedGpuIds === null) {
    return [];
  }
  return devices.filter(
    (device) =>
      device.indexKind === "physical" && selectedGpuIds.includes(device.index),
  );
}

/** DDP's global batch scales with its selected workers; other modes use one. */
export function trainingPreviewGpuCount(
  mode: TrainingParallelismMode,
  selectedGpuIds: number[] | null,
): number {
  return Math.max(mode === "ddp" ? (selectedGpuIds?.length ?? 0) : 1, 1);
}

export type RefreshedGpuValidation<T> =
  | { kind: "inputs-changed" }
  | { kind: "invalid"; errorKey: TranslationKey }
  | { kind: "valid"; availableGpuIds: number[] | null; config: T };

/** Refresh device membership and validate the same config snapshot before launch proceeds. */
export async function validateRefreshedTrainingGpuSelection<T>(options: {
  refresh: () => Promise<number[] | null>;
  abortIfInputsChanged: () => boolean;
  getConfig: () => T;
  validate: (
    config: T,
    availableGpuIds: number[] | null,
  ) => { ok: true } | { ok: false; errorKey: TranslationKey };
}): Promise<RefreshedGpuValidation<T>> {
  const availableGpuIds = await options.refresh();
  if (options.abortIfInputsChanged()) {
    return { kind: "inputs-changed" };
  }
  const config = options.getConfig();
  const validation = options.validate(config, availableGpuIds);
  if (!validation.ok) {
    return { kind: "invalid", errorKey: validation.errorKey };
  }
  return { kind: "valid", availableGpuIds, config };
}

/** Preserve selections during discovery; normalize stale selections once it settles. */
export function reconcileTrainingGpuSelection(
  mode: TrainingParallelismMode,
  ids: number[] | null,
  availableIds: number[] | null,
): {
  parallelismMode: TrainingParallelismMode;
  selectedGpuIds: number[] | null;
} {
  if (availableIds === null) {
    return { parallelismMode: mode, selectedGpuIds: ids };
  }
  const uniqueIds = [...new Set(ids ?? [])];
  const kept = uniqueIds.filter(
    (id) => Number.isSafeInteger(id) && id >= 0 && availableIds.includes(id),
  );
  // Keep an in-progress manual selection so validation can explain its arity.
  // Preserve an incomplete manual choice so validation can explain its arity.
  if (mode !== "auto" && ids !== null && kept.length === uniqueIds.length) {
    return { parallelismMode: mode, selectedGpuIds: ids };
  }
  const valid = mode === "single" ? kept.length === 1 : kept.length >= 2;
  if (mode === "auto" || !valid) {
    return { parallelismMode: "auto", selectedGpuIds: null };
  }
  return { parallelismMode: mode, selectedGpuIds: kept };
}
