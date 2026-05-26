// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type TrainingResourceKind = "model" | "dataset";
export type TrainingResourceNoticeStatus = "download" | "partial";

export interface TrainingResourceNotice {
  kind: TrainingResourceKind;
  status: TrainingResourceNoticeStatus;
  id: string;
  title: string;
  description: string;
}

export interface TrainingResourceNoticeInput {
  kind: TrainingResourceKind;
  id: string | null;
  isLocal: boolean;
  knownCached: boolean;
  localPath: string | null;
  completeSet: ReadonlySet<string>;
  partialSet: ReadonlySet<string>;
}

export interface ResourceInventoryRow {
  repoId: string | null;
  partial?: boolean;
}

export function completeResourceSet(
  rows: ReadonlyArray<ResourceInventoryRow>,
): Set<string> {
  const complete = new Set<string>();
  for (const row of rows) {
    if (!row.repoId || row.partial) continue;
    complete.add(row.repoId.toLowerCase());
  }
  return complete;
}

export function resolveTrainingResourceNotice({
  kind,
  id,
  isLocal,
  knownCached,
  localPath,
  completeSet,
  partialSet,
}: TrainingResourceNoticeInput): TrainingResourceNotice | null {
  const resourceId = id?.trim();
  if (!resourceId || isLocal) return null;

  const key = resourceId.toLowerCase();
  const isPartial = partialSet.has(key) && !completeSet.has(key);
  if (isPartial) {
    const noun = kind === "model" ? "Model" : "Dataset";
    return {
      kind,
      status: "partial",
      id: resourceId,
      title: `${noun} download will continue`,
      description:
        kind === "model"
          ? "Training will complete the partial model download before loading it."
          : "Training will complete the partial dataset download before reading it.",
    };
  }

  const isComplete =
    completeSet.has(key) || (knownCached && !partialSet.has(key));
  if (isComplete || localPath) return null;

  const noun = kind === "model" ? "Model" : "Dataset";
  return {
    kind,
    status: "download",
    id: resourceId,
    title: `${noun} will download at start`,
    description:
      kind === "model"
        ? "This model is not on this device yet. Training will download it automatically."
        : "This dataset is not on this device yet. Training will download it automatically.",
  };
}
