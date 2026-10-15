// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";
import type { TrainingParallelismMode } from "../types/config";

import type { SystemGpuDevice } from "@/hooks/gpu-selection";

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

// The server runs DDP for text models only; the other training paths are not rank-aware.
export function ddpSupportsModel(config: {
  modelType: string | null;
  isEmbeddingModel: boolean;
  isVisionModel: boolean;
  isAudioModel: boolean;
  isDatasetImage: boolean | null;
  isDatasetAudio: boolean;
}): boolean {
  return (
    (config.modelType === null || config.modelType === "text") &&
    !config.isEmbeddingModel &&
    !config.isVisionModel &&
    !config.isAudioModel &&
    !config.isDatasetImage &&
    !config.isDatasetAudio
  );
}

export function showsTrainingHardwareTab(
  mode: TrainingParallelismMode,
  devices: readonly SystemGpuDevice[],
): boolean {
  return (
    mode !== "auto" ||
    devices.filter((device) => device.indexKind === "physical").length >= 2
  );
}

export function trainingPreviewGpuCount(
  mode: TrainingParallelismMode,
  selectedGpuIds: number[] | null,
): number {
  return Math.max(mode === "ddp" ? (selectedGpuIds?.length ?? 0) : 1, 1);
}

export type RefreshedGpuValidation<T> =
  | { kind: "inputs-changed" }
  | { kind: "invalid"; errorKey: TranslationKey }
  | { kind: "valid"; availableGpuIds: number[] | null | undefined; config: T };

export async function validateRefreshedTrainingGpuSelection<T>(options: {
  refresh: (() => Promise<number[] | null>) | null;
  abortIfInputsChanged: () => boolean;
  getConfig: () => T;
  validate: (
    config: T,
    availableGpuIds: number[] | null | undefined,
  ) => { ok: true } | { ok: false; errorKey: TranslationKey };
}): Promise<RefreshedGpuValidation<T>> {
  const availableGpuIds =
    options.refresh === null ? undefined : await options.refresh();
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
  if (mode !== "auto" && ids !== null && kept.length === uniqueIds.length) {
    return { parallelismMode: mode, selectedGpuIds: ids };
  }
  const valid = mode === "single" ? kept.length === 1 : kept.length >= 2;
  if (mode === "auto" || !valid) {
    return { parallelismMode: "auto", selectedGpuIds: null };
  }
  return { parallelismMode: mode, selectedGpuIds: kept };
}
