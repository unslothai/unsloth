// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";

export function resolveStartTrainingButtonLabelKey({
  stopRequested,
  scheduled,
  scheduling,
  startBlocked,
  isLoadingModel,
  isCheckingDataset,
  hasModel,
  hasDataset,
  scheduleMode,
}: {
  stopRequested: boolean;
  scheduled: boolean;
  scheduling: boolean;
  startBlocked: boolean;
  isLoadingModel: boolean;
  isCheckingDataset: boolean;
  hasModel: boolean;
  hasDataset: boolean;
  scheduleMode: boolean;
}): TranslationKey {
  if (stopRequested) {
    return "studio.training.stopping";
  }
  if (scheduled) {
    return "studio.schedule.scheduled";
  }
  if (scheduling) {
    return "studio.schedule.scheduling";
  }
  if (startBlocked) {
    return "studio.training.starting";
  }
  if (isLoadingModel) {
    return "studio.training.loadingModel";
  }
  if (isCheckingDataset) {
    return "studio.training.checkingDataset";
  }
  if (!(hasModel || hasDataset)) {
    return "studio.training.chooseModelAndDataset";
  }
  if (!hasModel) {
    return "studio.training.chooseModel";
  }
  if (!hasDataset) {
    return "studio.training.chooseDataset";
  }
  return scheduleMode
    ? "studio.schedule.scheduleTraining"
    : "studio.training.startTraining";
}
