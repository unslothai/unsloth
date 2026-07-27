// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { LruMap } from "@/features/hub/lib/lru-map";
import { fetchWithTimeout } from "@/features/hub/lib/network";
import { useOnlineStatus } from "@/features/hub/hooks/use-online-status";

type AvatarCacheEntry =
  | { kind: "url"; url: string; expiresAt: number }
  | { kind: "miss-permanent" }
  | { kind: "miss-transient"; until: number; failures: number };

// Avatars rarely change; after this TTL the cached URL is still shown, then
// refreshed in the background and swapped if changed (stale-while-revalidate).
const URL_TTL_MS = 24 * 60 * 60 * 1000;

// Transient (429/5xx) avatar failures are negative-cached so they auto-recover.
// Each consecutive failure doubles the retry delay up to a cap (success resets);
// the backoff lives in the cache entry to survive virtualized-list remount churn.
const TRANSIENT_MISS_BASE_TTL_MS = 60_000;
const TRANSIENT_MISS_MAX_TTL_MS = 30 * 60_000;

// Bound each fetch so a stalled connection can't hold a concurrency permit open
// forever (which would eventually hang every queued lookup); timeout is treated
// as a transient miss so it auto-recovers.
const AVATAR_FETCH_TIMEOUT_MS = 10_000;

// Defer the first network lookup so rows scrolled past in under this window
// cancel before they ever touch HF, instead of a fast scroll firing hundreds of
// requests that swamp the gate. Cached avatars (url / miss) still resolve sync.
const AVATAR_FETCH_DEBOUNCE_MS = 200;

const cache = new LruMap<string, AvatarCacheEntry>(256);
const inflight = new Map<string, Promise<string | null>>();

// Cap concurrent avatar lookups so a fast scroll doesn't burst past the rate
// limit, mirroring the modelInfo limiter in hf-cache.ts.
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

// Expired transient misses report "no entry" so the caller refetches, but the
// entry is kept so its failure count can escalate the next backoff.
function readCache(name: string): AvatarCacheEntry | null {
  const entry = cache.get(name);
  if (!entry) return null;
  if (entry.kind === "miss-transient" && Date.now() >= entry.until) {
    return null;
  }
  return entry;
}

function readCachedUrl(name: string): string | null {
  if (!name) return null;
  const entry = readCache(name);
  return entry?.kind === "url" ? entry.url : null;
}

function transientMiss(name: string): AvatarCacheEntry {
  const prev = cache.get(name);
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
    `https://huggingface.co/api/organizations/${encodeURIComponent(name)}/overview`,
    `https://huggingface.co/api/users/${encodeURIComponent(name)}/overview`,
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
            : `https://huggingface.co${data.avatarUrl}`;
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
  const existing = inflight.get(name);
  if (existing) return existing;
  const promise = acquire()
    .then(() => fetchAvatarUrl(name))
    .finally(release)
    .then(
      ({ url, transient }) => {
        if (url) {
          cache.set(name, { kind: "url", url, expiresAt: Date.now() + URL_TTL_MS });
        } else if (transient) {
          cache.set(name, transientMiss(name));
        } else {
          cache.set(name, { kind: "miss-permanent" });
        }
        inflight.delete(name);
        return url;
      },
      () => {
        cache.set(name, transientMiss(name));
        inflight.delete(name);
        return null;
      },
    );
  inflight.set(name, promise);
  return promise;
}

export function useHfOwnerAvatar(
  owner: string | null | undefined,
  enabled = true,
): string | null {
  const key = owner?.trim() ?? "";
  const online = useOnlineStatus();
  const [state, setState] = useState<{ key: string; url: string | null }>(() => {
    return { key, url: readCachedUrl(key) };
  });
  const url = state.key === key ? state.url : readCachedUrl(key);

  useEffect(() => {
    // When disabled (virtualized list rows), never hit the network: show a
    // cached avatar if one exists, else the colored-initial tile. Keeps the
    // "All publishers" feed from firing a per-row lookup storm.
    if (!key || !online || !enabled) return;
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
      // Uncached: defer the network lookup so rows scrolled past in under
      // AVATAR_FETCH_DEBOUNCE_MS cancel before they ever hit HF.
      fetchTimer = setTimeout(runFetch, AVATAR_FETCH_DEBOUNCE_MS);
    };

    void attempt();

    return () => {
      cancelled = true;
      if (retryTimer != null) clearTimeout(retryTimer);
      if (fetchTimer != null) clearTimeout(fetchTimer);
    };
  }, [key, online, enabled]);

  return url;
}
