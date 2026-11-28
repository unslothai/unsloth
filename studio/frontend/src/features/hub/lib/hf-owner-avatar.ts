// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { getHfEndpoint, useHfEndpoint } from "@/lib/hf-endpoint";
import { LruMap } from "@/features/hub/lib/lru-map";
import { fetchWithTimeout } from "@/features/hub/lib/network";
import { useOnlineStatus } from "@/features/hub/hooks/use-online-status";

type AvatarCacheEntry =
  | { kind: "url"; url: string; expiresAt: number }
  | { kind: "miss-permanent"; until: number }
  | { kind: "miss-transient"; until: number; failures: number };

// Stale-while-revalidate after this TTL.
const URL_TTL_MS = 24 * 60 * 60 * 1000;

// Transient misses back off exponentially; the count lives in the entry to survive remounts.
const TRANSIENT_MISS_BASE_TTL_MS = 60_000;
const TRANSIENT_MISS_MAX_TTL_MS = 30 * 60_000;

// Bound each fetch so a stalled connection cannot hold a concurrency permit forever.
const AVATAR_FETCH_TIMEOUT_MS = 10_000;

// Debounced so rows scrolled past quickly never hit HF.
const AVATAR_FETCH_DEBOUNCE_MS = 200;

const cache = new LruMap<string, AvatarCacheEntry>(256);
const inflight = new Map<string, Promise<string | null>>();
const listeners = new Map<string, Set<() => void>>();
const persisted = new Map<string, PersistedAvatarEntry>();

const STORAGE_KEY = "unsloth:hf-owner-avatars:v1";
const MAX_PERSISTED_AVATARS = 256;

// A 404 owner is persisted too, or every launch re-asks HF twice per such owner.
const MISS_PERSIST_TTL_MS = 7 * 24 * 60 * 60 * 1000;

// url null = both profile endpoints 404'd; held until expiresAt.
interface PersistedAvatarEntry {
  key: string;
  url: string | null;
  expiresAt: number;
}

function subscribeAvatarKey(key: string, listener: () => void): () => void {
  let set = listeners.get(key);
  if (!set) {
    set = new Set();
    listeners.set(key, set);
  }
  set.add(listener);
  return () => {
    const current = listeners.get(key);
    if (!current) return;
    current.delete(listener);
    if (current.size === 0) listeners.delete(key);
  };
}

function notifyAvatarKey(key: string): void {
  const set = listeners.get(key);
  if (!set) return;
  for (const listener of set) listener();
}

function readStorage(): Storage | null {
  try {
    return typeof globalThis.localStorage !== "undefined"
      ? globalThis.localStorage
      : null;
  } catch {
    return null;
  }
}

function hydratePersistedAvatars(): void {
  const storage = readStorage();
  if (!storage) return;
  try {
    const raw = storage.getItem(STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return;
    for (const item of parsed) {
      const entry = (item ?? {}) as Partial<PersistedAvatarEntry>;
      const { key, url, expiresAt } = entry;
      if (typeof key !== "string" || typeof expiresAt !== "number") continue;
      if (url === null) {
        if (expiresAt <= Date.now()) continue;
        persisted.set(key, { key, url, expiresAt });
        cache.set(key, { kind: "miss-permanent", until: expiresAt });
      } else if (typeof url === "string") {
        persisted.set(key, { key, url, expiresAt });
        cache.set(key, { kind: "url", url, expiresAt });
      }
    }
  } catch {
    // Ignore corrupt or inaccessible storage.
  }
}

function persistAvatar(
  key: string,
  url: string | null,
  expiresAt: number,
): void {
  persisted.delete(key);
  persisted.set(key, { key, url, expiresAt });
  while (persisted.size > MAX_PERSISTED_AVATARS) {
    const oldest = persisted.keys().next().value;
    if (oldest === undefined) break;
    persisted.delete(oldest);
  }
  writePersisted();
}

function writePersisted(): void {
  try {
    readStorage()?.setItem(STORAGE_KEY, JSON.stringify([...persisted.values()]));
  } catch {
    // Ignore quota or private-mode storage errors.
  }
}

hydratePersistedAvatars();

/** Re-hydrate from storage, as a relaunch would. */
export function resetAvatarCacheForTests(): void {
  cache.clear();
  inflight.clear();
  persisted.clear();
  hydratePersistedAvatars();
}

// Mirrors the modelInfo limiter in hf-cache.ts.
const MAX_AVATAR_CONCURRENT = 6;
let activeFetches = 0;
const waiting: Array<() => void> = [];

function acquire(): Promise<void> {
  if (activeFetches < MAX_AVATAR_CONCURRENT) {
    activeFetches++;
    return Promise.resolve();
  }
  return new Promise<void>((resolve) =>
    waiting.push(() => {
      activeFetches++;
      resolve();
    }),
  );
}

function release(): void {
  activeFetches--;
  waiting.shift()?.();
}

// Keyed by endpoint and lowercased owner; hits and 404s outlive a late mirror switch.
function avatarKey(name: string): string {
  return `${getHfEndpoint()}::${name.trim().toLowerCase()}`;
}

// Expired misses read as absent; transient ones stay cached to escalate the next backoff.
function readCache(name: string): AvatarCacheEntry | null {
  const entry = cache.get(avatarKey(name));
  if (!entry) return null;
  if (entry.kind !== "url" && Date.now() >= entry.until) {
    return null;
  }
  return entry;
}

export function readCachedHfOwnerAvatar(name: string): string | null {
  if (!name.trim()) return null;
  const entry = readCache(name);
  return entry?.kind === "url" ? entry.url : null;
}

/** An <img> failed: expire every owner cached with that URL so the next mount asks HF again. */
export function expireHfOwnerAvatarUrl(url: string): void {
  // The cache and the persisted list evict in different orders, so walk both.
  const keys = [...cache.entries()]
    .filter(([, entry]) => entry.kind === "url" && entry.url === url)
    .map(([key]) => key);
  for (const key of keys) cache.set(key, { kind: "url", url, expiresAt: 0 });
  let changed = false;
  for (const entry of persisted.values()) {
    if (entry.url !== url || entry.expiresAt === 0) continue;
    entry.expiresAt = 0;
    changed = true;
  }
  if (changed) writePersisted();
}

function transientMiss(name: string): AvatarCacheEntry {
  const prev = cache.get(avatarKey(name));
  const failures = prev?.kind === "miss-transient" ? prev.failures + 1 : 1;
  const ttl = Math.min(
    TRANSIENT_MISS_BASE_TTL_MS * 2 ** (failures - 1),
    TRANSIENT_MISS_MAX_TTL_MS,
  );
  return { kind: "miss-transient", until: Date.now() + ttl, failures };
}

async function fetchAvatarUrl(
  name: string,
): Promise<{ url: string | null; transient: boolean }> {
  const candidates = [
    `${getHfEndpoint()}/api/organizations/${encodeURIComponent(name)}/overview`,
    `${getHfEndpoint()}/api/users/${encodeURIComponent(name)}/overview`,
  ];

  let sawTransient = false;
  for (const url of candidates) {
    try {
      const res = await fetchWithTimeout(
        url,
        {
          credentials: "omit",
        },
        AVATAR_FETCH_TIMEOUT_MS,
      );
      if (res.ok) {
        const data = (await res.json()) as { avatarUrl?: string };
        if (data.avatarUrl) {
          const resolved = data.avatarUrl.startsWith("http")
            ? data.avatarUrl
            : `${getHfEndpoint()}${data.avatarUrl}`;
          return { url: resolved, transient: false };
        }
        continue;
      }
      if (res.status === 404) {
        continue;
      }
      sawTransient = true;
    } catch {
      sawTransient = true;
    }
  }
  return { url: null, transient: sawTransient };
}

function loadAvatar(name: string): Promise<string | null> {
  const key = avatarKey(name);
  const existing = inflight.get(key);
  if (existing) return existing;
  const promise = acquire()
    .then(() => fetchAvatarUrl(name))
    .finally(release)
    .then(
      ({ url, transient }) => {
        if (url) {
          const expiresAt = Date.now() + URL_TTL_MS;
          cache.set(key, { kind: "url", url, expiresAt });
          persistAvatar(key, url, expiresAt);
          notifyAvatarKey(key);
        } else if (transient) {
          cache.set(key, transientMiss(name));
        } else {
          const until = Date.now() + MISS_PERSIST_TTL_MS;
          cache.set(key, { kind: "miss-permanent", until });
          persistAvatar(key, null, until);
        }
        inflight.delete(key);
        return url;
      },
      () => {
        cache.set(key, transientMiss(name));
        inflight.delete(key);
        return null;
      },
    );
  inflight.set(key, promise);
  return promise;
}

export function useHfOwnerAvatar(
  owner: string | null | undefined,
  enabled = true,
): string | null {
  const key = owner?.trim() ?? "";
  const online = useOnlineStatus();
  const hfEndpoint = useHfEndpoint();
  const [state, setState] = useState<{ key: string; url: string | null }>(() => {
    return { key, url: readCachedHfOwnerAvatar(key) };
  });
  const url = state.key === key ? state.url : readCachedHfOwnerAvatar(key);

  useEffect(() => {
    if (!key) return;
    // Subscribed even when disabled, so remote={false} rows pick up a logo another view fetched.
    const syncFromCache = () => {
      const next = readCachedHfOwnerAvatar(key);
      setState((prev) =>
        prev.key === key && prev.url === next ? prev : { key, url: next },
      );
    };
    syncFromCache();
    const unsubscribe = subscribeAvatarKey(avatarKey(key), syncFromCache);

    // Disabled (virtualized rows): never hit the network, to avoid a per-row lookup storm.
    if (!online || !enabled) {
      return unsubscribe;
    }
    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let fetchTimer: ReturnType<typeof setTimeout> | null = null;

    const scheduleRetry = (until: number) => {
      const wait = Math.max(until - Date.now(), 0) + 100;
      retryTimer = setTimeout(() => {
        if (!cancelled) void attempt();
      }, wait);
    };

    const runFetch = () => {
      void loadAvatar(key).then((next) => {
        if (cancelled) return;
        setState({ key, url: next });
        if (next == null) {
          const post = readCache(key);
          if (post?.kind === "miss-transient") {
            scheduleRetry(post.until);
          }
        }
      });
    };

    const attempt = async () => {
      const cached = readCache(key);
      if (cached?.kind === "url") {
        if (!cancelled) setState({ key, url: cached.url });
        if (cached.expiresAt <= Date.now()) {
          void loadAvatar(key).then((next) => {
            if (!cancelled && next) setState({ key, url: next });
          });
        }
        return;
      }
      if (cached?.kind === "miss-permanent") {
        if (!cancelled) setState({ key, url: null });
        return;
      }
      if (cached?.kind === "miss-transient") {
        if (!cancelled) setState({ key, url: null });
        scheduleRetry(cached.until);
        return;
      }
      fetchTimer = setTimeout(runFetch, AVATAR_FETCH_DEBOUNCE_MS);
    };

    void attempt();

    return () => {
      cancelled = true;
      unsubscribe();
      if (retryTimer != null) clearTimeout(retryTimer);
      if (fetchTimer != null) clearTimeout(fetchTimer);
    };
  }, [key, online, enabled, hfEndpoint]);

  return url;
}
