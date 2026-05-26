// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { modelIdsMatch } from "../../../lib/model-identity.ts";
import type { TrainingConfigState } from "../../training/types/config.ts";
import type { SelectedModelView } from "../types";

export type TrainingSelectionConflictSnapshot = {
  kind: "model" | "dataset";
  id: string;
  source: "model" | "huggingface" | "upload";
};

export type TrainingSelectionTarget = {
  kind: "model" | "dataset";
  id: string | null;
  source: "model" | "huggingface" | "upload";
};

export function createTrainingSelectionTarget(
  target: SelectedModelView,
  isDatasetMode: boolean,
): TrainingSelectionTarget {
  const { resource } = target;
  const repoId = resource.repoId ?? target.hubRepoId ?? target.id;
  if (isDatasetMode) {
    const isLocal = resource.cacheState === "local" && !!resource.localPath;
    return {
      kind: "dataset",
      id: isLocal ? resource.localPath : repoId,
      source: isLocal ? "upload" : "huggingface",
    };
  }
  return {
    kind: "model",
    id: resource.trainId,
    source: "model",
  };
}

export function getTrainingSelectionConflict(
  state: Pick<
    TrainingConfigState,
    "dataset" | "datasetSource" | "selectedModel" | "uploadedFile"
  >,
  target: TrainingSelectionTarget,
): TrainingSelectionConflictSnapshot | null {
  if (!target.id) {
    return null;
  }
  if (target.kind === "dataset") {
    const currentId =
      state.datasetSource === "upload" ? state.uploadedFile : state.dataset;
    if (!currentId) {
      return null;
    }
    if (
      state.datasetSource === target.source &&
      modelIdsMatch(currentId, target.id)
    ) {
      return null;
    }
    return {
      kind: "dataset",
      id: currentId,
      source: state.datasetSource,
    };
  }
  if (!state.selectedModel || modelIdsMatch(state.selectedModel, target.id)) {
    return null;
  }
  return {
    kind: "model",
    id: state.selectedModel,
    source: "model",
  };
}

export function trainingSelectionConflictEqual(
  a: TrainingSelectionConflictSnapshot | null,
  b: TrainingSelectionConflictSnapshot | null,
): boolean {
  return a?.kind === b?.kind && a?.source === b?.source && a?.id === b?.id;
}
