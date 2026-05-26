// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { LruMap } from "@/lib/lru-map";
import { fetchWithTimeout } from "@/lib/network";
import { useOnlineStatus } from "@/hooks";

type AvatarCacheEntry =
  | { kind: "url"; url: string; expiresAt: number }
  | { kind: "miss-permanent" }
  | { kind: "miss-transient"; until: number; failures: number };

// Avatars rarely change, but "never" is wrong for a session that stays open for
// days. After this the cached URL is still shown immediately, then refreshed in
// the background and swapped if it changed (stale-while-revalidate).
const URL_TTL_MS = 24 * 60 * 60 * 1000;

// Fast scrolling through the discover list can fire dozens of avatar lookups
// at once. HF will sometimes 429 / 5xx those bursts; those failures used to
// pin the owner avatar to null until the user refreshed the page. We now
// negative-cache transient failures so they auto-recover, but a persistently
// flapping owner must not refetch on a fixed cadence for the list's whole
// lifetime: each consecutive failure doubles the retry delay up to a cap, and
// a success resets it. The backoff lives in the cache entry so it survives the
// remount churn of a virtualized list.
const TRANSIENT_MISS_BASE_TTL_MS = 60_000;
const TRANSIENT_MISS_MAX_TTL_MS = 30 * 60_000;

// A stalled connection (captive portal, hung socket) must never hold a
// concurrency permit open: without a deadline the fetch never settles, the
// permit is never released, and once MAX_AVATAR_CONCURRENT requests pile up
// every queued lookup hangs until a full page reload. Abort and treat the
// timeout as a transient miss so it auto-recovers.
const AVATAR_FETCH_TIMEOUT_MS = 10_000;

const cache = new LruMap<string, AvatarCacheEntry>(256);
const inflight = new Map<string, Promise<string | null>>();

// Cap how many avatar lookups hit HF at once so a fast scroll doesn't burst
// past the rate limit, mirroring the modelInfo limiter in hf-cache.ts.
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

// An expired transient miss is reported as "no entry" so the caller refetches,
// but the entry is kept so its failure count can escalate the next backoff.
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

// Manual cache-bust path: drop a specific owner (or all) so the next render
// refetches. Call after an action that's known to change an avatar.
export function invalidateOwnerAvatar(owner?: string): void {
  if (owner) cache.delete(owner.trim());
  else cache.clear();
}

export function useHfOwnerAvatar(owner: string | null | undefined): string | null {
  const key = owner?.trim() ?? "";
  const online = useOnlineStatus();
  const [state, setState] = useState<{ key: string; url: string | null }>(() => {
    return { key, url: readCachedUrl(key) };
  });
  const url = state.key === key ? state.url : readCachedUrl(key);

  useEffect(() => {
    if (!key || !online) return;
    let cancelled = false;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    const scheduleRetry = (until: number) => {
      const wait = Math.max(until - Date.now(), 0) + 100;
      retryTimer = setTimeout(() => {
        if (!cancelled) void attempt();
      }, wait);
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
      const next = await loadAvatar(key);
      if (cancelled) return;
      setState({ key, url: next });
      if (next == null) {
        const post = readCache(key);
        if (post?.kind === "miss-transient") {
          scheduleRetry(post.until);
        }
      }
    };

    void attempt();

    return () => {
      cancelled = true;
      if (retryTimer != null) clearTimeout(retryTimer);
    };
  }, [key, online]);

  return url;
}
