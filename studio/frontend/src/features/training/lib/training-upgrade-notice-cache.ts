// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TransformersUpgradeCheck } from "@/features/transformers-upgrade";

// One answer per (sidecar generation, model, cached copy, token) for the Configure
// preview. The config.json read behind the check is a network round trip and the notice
// hook runs on every Configure render, so without this, scrubbing a slider would re-ask.
//
// The generation is in the key because an install invalidates every answer here: the
// release offered is now installed, and the sidecar it activated flips forces16Bit on
// for models that reported 4-bit before.
const cache = new Map<string, TransformersUpgradeCheck>();

// Separator no field can contain: a model id, a cache path and a token are all printable.
const KEY_SEPARATOR = "\u0001";

let cachedGeneration = 0;

/** Key one preview answer. `sidecarGeneration` counts completed installs. */
export function upgradeNoticeCacheKey(
  sidecarGeneration: number,
  modelName: string,
  preferLocalCache: boolean,
  localPath: string | null,
  hfToken: string,
): string {
  // preferLocalCache is its own field: a known-cached row with a null path resolves to a
  // pinned snapshot, a different answer from the same model asked without the preference.
  return [
    sidecarGeneration,
    modelName,
    preferLocalCache ? "1" : "0",
    localPath ?? "",
    hfToken,
  ].join(KEY_SEPARATOR);
}

// Entries from a superseded generation can never be read again, so drop them rather than
// let an install-per-session leak the whole map.
//
// The generation only counts up, so an older one is always a straggler: a check fired
// before an install can still be in flight when it lands, resolving after the
// post-install check answered. Letting that write rewind the generation would clear the
// fresh entry and store the stale answer in its place, and since its effect was already
// cleaned up nothing re-asks, so the preview would show no notice for the rest of the
// session. Superseded reads and writes are ignored: null, never the whole map.
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
