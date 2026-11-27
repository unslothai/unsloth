// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type StagedDims = {
  key: string;
  contextLength: number | null;
  layerCount: number | null;
  moeLayerCount: number | null;
  isDiffusion?: boolean;
  diffusionUnknown?: boolean;
};

const STAGED_DIMS_CACHE_LIMIT = 32;

// Collapsing the Model section unmounts the config page; without this every reopen re-reads the GGUF header first.
const stagedDimsCache = new Map<string, StagedDims>();

function conclusive(dims: StagedDims): boolean {
  return !dims.diffusionUnknown && (dims.contextLength != null || dims.layerCount != null);
}

/** Only a header that was actually read: an undownloaded GGUF answers no dims and must re-probe after download. */
export function cachedStagedDims(key: string | null): StagedDims | null {
  const dims = key == null ? undefined : stagedDimsCache.get(key);
  return dims && conclusive(dims) ? dims : null;
}

/** The last probe for this key read no header, so no estimate followed: do not hold a row for it. */
export function stagedDimsLastInconclusive(key: string | null): boolean {
  const dims = key == null ? undefined : stagedDimsCache.get(key);
  return dims != null && !conclusive(dims);
}

export function rememberStagedDims(dims: StagedDims): void {
  stagedDimsCache.delete(dims.key);
  stagedDimsCache.set(dims.key, dims);
  if (stagedDimsCache.size > STAGED_DIMS_CACHE_LIMIT) {
    stagedDimsCache.delete(stagedDimsCache.keys().next().value as string);
  }
}
