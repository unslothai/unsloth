


interface ResumeModelCacheFields {
  actualModelRepoId?: string | null;
  modelKnownCached?: boolean;
  modelLocalPath?: string | null;
  modelSnapshotPath?: string | null;
}

export function resolveResumeRemoteCodeCache({
  actualModelRepoId,
  modelKnownCached,
  modelLocalPath,
  modelSnapshotPath,
}: ResumeModelCacheFields): {
  preferLocalCache: boolean;
  modelLocalPath: string | null;
  modelSnapshotPath: string | null;
  modelSnapshotRepoId: string | null;
} {
  const resolvedLocalPath = modelSnapshotPath || modelLocalPath || null;
  return {
    preferLocalCache: Boolean(resolvedLocalPath || modelKnownCached),
    modelLocalPath: resolvedLocalPath,
    modelSnapshotPath: modelSnapshotPath || null,
    modelSnapshotRepoId: modelSnapshotPath ? actualModelRepoId || null : null,
  };
}
