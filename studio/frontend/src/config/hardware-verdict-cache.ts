// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The last hardware verdict the backend actually measured, kept across reloads so a
// returning user does not paint on the browser-platform guess. Its own import-free module
// so it is testable: env.ts reaches import.meta.env through api-base.ts, which only vite
// can load.

export type CachedVerdict = {
  deviceType: string;
  chatOnly: boolean;
  chatOnlyReason: string | null;
};

// Versioned in the key rather than inside the record: a schema change simply picks a new
// key, so an old build downgraded onto a new cache reads nothing instead of a shape it
// does not understand, and the stale record is overwritten on the next settled reply.
export const VERDICT_CACHE_KEY = "unsloth_hardware_verdict_v1";

type StorageLike = {
  getItem: (key: string) => string | null;
  setItem: (key: string, value: string) => void;
};

// Serialized with the fields in a fixed order, so an unchanged verdict compares equal as a
// string and cacheSettledVerdict can skip the write. The sidebar re-reads /api/health every
// 15s while a chat-only host waits on its MLX self-heal, and each identical write would
// otherwise fire a `storage` event in every other open tab.
function serialize(verdict: CachedVerdict): string {
  return JSON.stringify({
    deviceType: verdict.deviceType,
    chatOnly: verdict.chatOnly,
    chatOnlyReason: verdict.chatOnlyReason,
  });
}

/** localStorage when it is usable. Safari with site data blocked throws on the property
 * access itself, not just on the call, so even reaching for it is guarded. */
function storage(): StorageLike | null {
  try {
    if (typeof localStorage === "undefined") return null;
    return localStorage;
  } catch {
    return null;
  }
}

function isCachedVerdict(value: unknown): value is CachedVerdict {
  if (typeof value !== "object" || value === null) return false;
  const record = value as Record<string, unknown>;
  return (
    typeof record.deviceType === "string" &&
    record.deviceType.length > 0 &&
    typeof record.chatOnly === "boolean" &&
    (record.chatOnlyReason === null || typeof record.chatOnlyReason === "string")
  );
}

/** The last settled verdict, or null when there is nothing usable to seed from.
 *
 * Anything unreadable -- absent, unparseable, or the wrong shape -- is null rather than a
 * throw or a partial record: the caller's fallback is the behaviour that shipped before
 * this cache existed, so a corrupt value costs a guess, never a broken load. */
export function readCachedVerdict(): CachedVerdict | null {
  const store = storage();
  if (!store) return null;
  let raw: string | null = null;
  try {
    raw = store.getItem(VERDICT_CACHE_KEY);
  } catch {
    return null;
  }
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isCachedVerdict(parsed)) return null;
    return {
      deviceType: parsed.deviceType,
      chatOnly: parsed.chatOnly,
      chatOnlyReason: parsed.chatOnlyReason,
    };
  } catch {
    return null;
  }
}

/** Record a verdict the backend has measured.
 *
 * Callers must never pass a provisional or deferred reply. Those carry the pre-detection
 * default, chat_only true, so caching one would seed the next load with "this machine
 * cannot train" -- the wrong answer this cache exists to stop showing. */
export function cacheSettledVerdict(verdict: CachedVerdict): void {
  const store = storage();
  if (!store) return;
  const next = serialize(verdict);
  try {
    if (store.getItem(VERDICT_CACHE_KEY) === next) return;
    store.setItem(VERDICT_CACHE_KEY, next);
  } catch {
    // A full or read-only quota is not worth failing a page load over.
  }
}
