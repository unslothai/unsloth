// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelInventoryFormat } from "../../inventory/types.ts";
import { looksLikeLocalPath } from "../../../lib/local-path.ts";
import {
  shouldLoadFromLocalFilesOnly,
  type LocalFilesOnlySelection,
} from "./local-files-only.ts";

export type ActiveModelLoadSource = LocalFilesOnlySelection & {
  localPath?: string | null;
  modelFormat?: ModelInventoryFormat | null;
  ggufVariant?: string | null;
  nativePathToken?: string | null;
};

export type RollbackLoadOptions = {
  localFilesOnly: boolean;
  localPath: string | null;
  modelFormat: ModelInventoryFormat | null;
};

function cleanedPath(value: string | null | undefined): string | null {
  return value?.trim() || null;
}

export function buildActiveModelLoadSource(
  source: ActiveModelLoadSource,
): ActiveModelLoadSource | null {
  const localPath = cleanedPath(source.localPath);
  const nativePathToken = cleanedPath(source.nativePathToken);
  const hasSource =
    source.source !== undefined ||
    source.isDownloaded !== undefined ||
    source.isPartial !== undefined ||
    source.preferLocalCache !== undefined ||
    localPath !== null ||
    source.modelFormat != null ||
    source.ggufVariant != null ||
    nativePathToken !== null;
  if (!hasSource) return null;

  return {
    source: source.source,
    isDownloaded: source.isDownloaded ?? false,
    isPartial: source.isPartial ?? false,
    preferLocalCache: source.preferLocalCache ?? false,
    localPath,
    modelFormat: source.modelFormat ?? null,
    ggufVariant: source.ggufVariant ?? null,
    nativePathToken,
  };
}

export function resolveRollbackLoadOptions({
  previousCheckpoint,
  previousVariant,
  previousIsLora,
  previousActiveNativePathToken,
  previousLoadSource,
}: {
  previousCheckpoint: string;
  previousVariant?: string | null;
  previousIsLora?: boolean | null;
  previousActiveNativePathToken?: string | null;
  previousLoadSource?: ActiveModelLoadSource | null;
}): RollbackLoadOptions {
  const nativePathToken =
    cleanedPath(previousLoadSource?.nativePathToken) ??
    cleanedPath(previousActiveNativePathToken);
  const localFilesOnly = shouldLoadFromLocalFilesOnly({
    modelId: previousCheckpoint,
    nativePathToken,
    isCachedLora: Boolean(previousIsLora) && looksLikeLocalPath(previousCheckpoint),
    selection: previousLoadSource,
  });

  return {
    localFilesOnly,
    localPath: localFilesOnly ? cleanedPath(previousLoadSource?.localPath) : null,
    modelFormat:
      previousLoadSource?.modelFormat ?? (previousVariant ? "gguf" : null),
  };
}
