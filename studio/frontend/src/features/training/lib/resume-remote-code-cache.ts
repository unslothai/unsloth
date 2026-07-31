// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

interface ResumeModelCacheFields {
  modelKnownCached?: boolean;
  modelLocalPath?: string | null;
  modelSnapshotPath?: string | null;
}

export function resolveResumeRemoteCodeCache({
  modelKnownCached,
  modelLocalPath,
  modelSnapshotPath,
}: ResumeModelCacheFields): {
  preferLocalCache: boolean;
  modelLocalPath: string | null;
} {
  const resolvedLocalPath =
    modelSnapshotPath || (modelKnownCached ? modelLocalPath : null) || null;
  return {
    preferLocalCache: Boolean(modelSnapshotPath || modelKnownCached),
    modelLocalPath: resolvedLocalPath,
  };
}
