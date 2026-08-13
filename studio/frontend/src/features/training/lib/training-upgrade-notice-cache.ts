// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TransformersUpgradeCheck } from "@/features/transformers-upgrade";

// One answer per (sidecar generation, model, cached copy, token) for the Configure
// preview. The backend caches the PyPI/GitHub snapshot for a day, but the config.json
// read behind it is still a network round trip and the notice hook runs on every
// Configure render; without this, scrubbing a slider would re-ask.
//
// The generation is in the key because an install invalidates every answer in here: the
// release the preview offered is now installed, and the sidecar it activated flips
// forces16Bit on for models that reported 4-bit before. A key without it leaves Configure
// promising "QLoRA - 4-bit" for a run the sidecar will load in 16-bit, which is the VRAM
// understatement this preview exists to prevent.
const cache = new Map<string, TransformersUpgradeCheck>();

// Separator no field can contain: a model id, a cache path and a token are all printable.
const KEY_SEPARATOR = "\u0001";

let cachedGeneration = 0;

/** Key one preview answer. `sidecarGeneration` counts completed installs. */
export function upgradeNoticeCacheKey(
  sidecarGeneration: number,
  modelName: string,
  localPath: string | null,
  hfToken: string,
): string {
  return [sidecarGeneration, modelName, localPath ?? "", hfToken].join(
    KEY_SEPARATOR,
  );
}

// Entries from a superseded generation can never be read again, so drop them rather
// than let an install-per-session leak the whole map.
function activeCache(sidecarGeneration: number): typeof cache {
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
  return activeCache(sidecarGeneration).get(key) ?? null;
}

export function hasUpgradeNoticeCache(
  sidecarGeneration: number,
  key: string,
): boolean {
  return activeCache(sidecarGeneration).has(key);
}

export function writeUpgradeNoticeCache(
  sidecarGeneration: number,
  key: string,
  check: TransformersUpgradeCheck,
): void {
  activeCache(sidecarGeneration).set(key, check);
}
