// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TransformersUpgradeCheck } from "@/features/transformers-upgrade";

// Avoids a network round trip per Configure render. Keyed on the sidecar generation because an
// install invalidates every answer.
const cache = new Map<string, TransformersUpgradeCheck>();

// Separator no field can contain: a model id, a cache path and a token are all printable.
const KEY_SEPARATOR = "\u0001";

let cachedGeneration = 0;

export function upgradeNoticeCacheKey(
  sidecarGeneration: number,
  modelName: string,
  preferLocalCache: boolean,
  localPath: string | null,
  hfToken: string,
): string {
  // preferLocalCache is its own field: a null-path known-cached row resolves to a pinned snapshot.
  return [
    sidecarGeneration,
    modelName,
    preferLocalCache ? "1" : "0",
    localPath ?? "",
    hfToken,
  ].join(KEY_SEPARATOR);
}

// Drop superseded generations. A straggler from before an install must not rewind the generation
// and replace the fresh answer, so superseded reads and writes return null.
function activeCache(sidecarGeneration: number): typeof cache | null {
  if (sidecarGeneration < cachedGeneration) {
    return null;
  }
  if (sidecarGeneration !== cachedGeneration) {
    cachedGeneration = sidecarGeneration;
    cache.clear();
  }
  return cache;
}

export function readUpgradeNoticeCache(
  sidecarGeneration: number,
  key: string,
): TransformersUpgradeCheck | null {
  return activeCache(sidecarGeneration)?.get(key) ?? null;
}

export function hasUpgradeNoticeCache(
  sidecarGeneration: number,
  key: string,
): boolean {
  return activeCache(sidecarGeneration)?.has(key) ?? false;
}

export function writeUpgradeNoticeCache(
  sidecarGeneration: number,
  key: string,
  check: TransformersUpgradeCheck,
): void {
  activeCache(sidecarGeneration)?.set(key, check);
}
