// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type TrainingResourceKind = "model" | "dataset";
export type TrainingResourceNoticeStatus = "download" | "partial";

export interface TrainingResourceNotice {
  kind: TrainingResourceKind;
  status: TrainingResourceNoticeStatus;
  id: string;
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
  rows: readonly ResourceInventoryRow[],
): Set<string> {
  const complete = new Set<string>();
  for (const row of rows) {
    if (!row.repoId || row.partial) {
      continue;
    }
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
  if (!resourceId || isLocal) {
    return null;
  }

  const key = resourceId.toLowerCase();
  const isPartial = partialSet.has(key) && !completeSet.has(key);
  if (isPartial) {
    return { kind, status: "partial", id: resourceId };
  }

  const isComplete =
    completeSet.has(key) || (knownCached && !partialSet.has(key));
  if (isComplete || localPath) {
    return null;
  }

  return { kind, status: "download", id: resourceId };
}
