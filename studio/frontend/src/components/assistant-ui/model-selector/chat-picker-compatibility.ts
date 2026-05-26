// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  CachedInventoryRow,
  LocalInventoryRow,
} from "../../../features/inventory/types";
import { classifyUnslothSupport } from "../../../lib/unsloth-support.ts";

function supportedForChat({
  modelId,
  pipelineTag,
  tags,
  libraryName,
  quantMethod,
  deviceType,
}: {
  modelId?: string | null;
  pipelineTag?: string | null;
  tags?: readonly string[] | null;
  libraryName?: string | null;
  quantMethod?: string | null;
  deviceType?: string | null;
}): boolean {
  return (
    classifyUnslothSupport({
      modelId,
      pipelineTag,
      tags,
      libraryName,
      quantMethod,
      deviceType,
    }).status !== "unsupported"
  );
}

export function cachedInventoryRowCanChat(
  row: CachedInventoryRow,
  deviceType?: string | null,
): boolean {
  if (row.partial || !row.capabilities.canChat) return false;
  return supportedForChat({
    modelId: row.repoId,
    pipelineTag: row.pipelineTag,
    tags: row.tags,
    libraryName: row.libraryName,
    quantMethod: row.quantMethod,
    deviceType,
  });
}

export function localInventoryRowCanChat(
  row: LocalInventoryRow,
  deviceType?: string | null,
): boolean {
  if (row.partial || !row.capabilities.canChat) return false;
  const modelId =
    row.repoId ??
    row.baseModelHubId ??
    (row.baseModelSource === "huggingface" ? row.baseModel : null);
  return supportedForChat({
    modelId,
    pipelineTag: row.pipelineTag,
    tags: row.tags,
    libraryName: row.libraryName,
    quantMethod: row.quantMethod,
    deviceType,
  });
}
