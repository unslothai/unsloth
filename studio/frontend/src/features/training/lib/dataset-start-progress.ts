// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type DatasetSelection = { hf_dataset?: string | null };

export function getTrainingDatasetRepositoryIds(request: {
  training_datasets?: DatasetSelection[] | null;
  hf_dataset?: string | null;
}): string[] {
  const repositoryIds = (request.training_datasets ?? [])
    .map((entry) => entry.hf_dataset?.trim())
    .filter((value): value is string => Boolean(value));
  if (repositoryIds.length > 0) return repositoryIds;
  const legacyRepositoryId = request.hf_dataset?.trim();
  return legacyRepositoryId ? [legacyRepositoryId] : [];
}

export function resolveActiveDataset(
  repositoryIds: string[],
  reportedIndex: number | null,
  reportedTotal: number | null,
  reportedRepositoryId: string | null,
): { repositoryId: string | null; index: number; total: number } {
  const total = reportedTotal ?? repositoryIds.length;
  const index = Math.min(Math.max(reportedIndex ?? 1, 1), Math.max(total, 1));
  return {
    repositoryId: reportedRepositoryId ?? repositoryIds[index - 1] ?? null,
    index,
    total,
  };
}
