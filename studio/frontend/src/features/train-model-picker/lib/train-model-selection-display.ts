


import { hubResourceIdsEqual } from "@/components/resource-picker/hub-resource-id";
import { pathDisplayName } from "@/components/resource-picker/path-display-name";
import {
  type CachedInventoryRow,
  type LocalInventoryRow,
  type LocalSource,
  type ModelInventoryFormat,
  repoOf,
} from "@/features/hub";
import {
  cacheLocalPathMatchesSelection,
  isLocalTrainingModelSelection,
} from "@/features/training";

export interface TrainModelDisplayCandidate {
  readonly id: string;
  readonly title: string;
  readonly source: LocalSource;
  readonly localPath: string | null;
  readonly modelFormat: ModelInventoryFormat | null;
}

export function trainModelDisplayCandidateMatchesSelection({
  candidate,
  selectedModel,
  selectedLocalPath,
  selectedFormat,
}: {
  candidate: TrainModelDisplayCandidate;
  selectedModel: string | null;
  selectedLocalPath: string | null;
  selectedFormat: ModelInventoryFormat | null;
}): boolean {
  if (
    !selectedModel ||
    (candidate.source === "hf_cache"
      ? !hubResourceIdsEqual(selectedModel, candidate.id)
      : selectedModel !== candidate.id)
  ) {
    return false;
  }
  if (
    selectedFormat &&
    candidate.modelFormat &&
    selectedFormat !== candidate.modelFormat
  ) {
    return false;
  }
  if (selectedLocalPath?.trim()) {
    return cacheLocalPathMatchesSelection(
      candidate.localPath,
      selectedLocalPath,
    );
  }
  return true;
}

export function trainModelSelectionDisplayName({
  selectedModel,
  knownCached,
  selectedLocalPath,
  selectedFormat,
  candidates,
}: {
  selectedModel: string | null;
  knownCached: boolean;
  selectedLocalPath: string | null;
  selectedFormat: ModelInventoryFormat | null;
  candidates: readonly TrainModelDisplayCandidate[];
}): string | null {
  if (!selectedModel) {
    return null;
  }
  if (
    !isLocalTrainingModelSelection({
      model: selectedModel,
      knownCached,
      localPath: selectedLocalPath,
    })
  ) {
    return repoOf(selectedModel);
  }
  const selectedTitle = candidates.find((candidate) =>
    trainModelDisplayCandidateMatchesSelection({
      candidate,
      selectedModel,
      selectedLocalPath,
      selectedFormat,
    }),
  )?.title;
  return (
    selectedTitle ?? pathDisplayName(selectedLocalPath?.trim() || selectedModel)
  );
}

export function toTrainModelDisplayCandidate(
  row: LocalInventoryRow,
): TrainModelDisplayCandidate {
  const cachedRepoId = row.source === "hf_cache" ? row.repoId?.trim() : null;
  return {
    id: cachedRepoId || row.loadId,
    title: row.repoId ?? row.title,
    source: row.source,
    localPath: row.path,
    modelFormat: row.modelFormat,
  };
}

export function toCachedTrainModelDisplayCandidate(
  row: CachedInventoryRow,
): TrainModelDisplayCandidate {
  return {
    id: row.repoId,
    title: row.repoId,
    source: "hf_cache",
    localPath: row.cachePath ?? null,
    modelFormat: row.modelFormat,
  };
}
