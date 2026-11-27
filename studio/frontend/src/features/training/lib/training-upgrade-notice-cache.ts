// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TransformersUpgradeCheck } from "@/features/transformers-upgrade";

// Avoids a round trip per render; keyed on sidecar generation, since install invalidates all.
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

// Old generations return null on read and write, so a straggler cannot rewind the generation.
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
