// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { looksLikeLocalPath } from "../../../lib/local-path.ts";
import { isHuggingFaceOffline } from "../../../lib/network.ts";

export type LocalModelLoadSource =
  | "hub"
  | "lora"
  | "exported"
  | "local"
  | "external";

export type LocalFilesOnlySelection = {
  source?: LocalModelLoadSource;
  isDownloaded?: boolean;
  isPartial?: boolean;
  preferLocalCache?: boolean;
};

export function shouldLoadFromLocalFilesOnly({
  modelId,
  nativePathToken,
  isCachedLora,
  selection,
}: {
  modelId: string;
  nativePathToken?: string | null;
  isCachedLora: boolean;
  selection?: LocalFilesOnlySelection | null;
}): boolean {
  if (isHuggingFaceOffline() || nativePathToken || looksLikeLocalPath(modelId)) {
    return true;
  }
  if (
    isCachedLora ||
    selection?.source === "local" ||
    selection?.source === "lora" ||
    selection?.source === "exported"
  ) {
    return true;
  }
  return (
    selection?.source === "hub" &&
    selection.preferLocalCache === true &&
    selection.isDownloaded === true &&
    selection.isPartial !== true
  );
}
