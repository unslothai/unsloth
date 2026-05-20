// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { LruMap } from "@/lib/lru-map";

type AvatarCacheEntry =
  | { kind: "url"; url: string }
  | { kind: "miss-permanent" }
  | { kind: "miss-transient"; until: number };

// Fast scrolling through the discover list can fire dozens of avatar lookups
// at once. HF will sometimes 429 / 5xx those bursts; those failures used to
// pin the owner avatar to null until the user refreshed the page. We now
// negative-cache transient failures with a short TTL so they auto-recover.
const TRANSIENT_MISS_TTL_MS = 60_000;

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

function readCache(name: string): AvatarCacheEntry | null {
  const entry = cache.get(name);
  if (!entry) return null;
  if (entry.kind === "miss-transient" && Date.now() >= entry.until) {
    cache.delete(name);
    return null;
  }
  return entry;
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
      const res = await fetch(url, {
        credentials: "omit",
        signal: AbortSignal.timeout(AVATAR_FETCH_TIMEOUT_MS),
      });
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
          cache.set(name, { kind: "url", url });
        } else if (transient) {
          cache.set(name, {
            kind: "miss-transient",
            until: Date.now() + TRANSIENT_MISS_TTL_MS,
          });
        } else {
          cache.set(name, { kind: "miss-permanent" });
        }
        inflight.delete(name);
        return url;
      },
      () => {
        cache.set(name, {
          kind: "miss-transient",
          until: Date.now() + TRANSIENT_MISS_TTL_MS,
        });
        inflight.delete(name);
        return null;
      },
    );
  inflight.set(name, promise);
  return promise;
}

export function useHfOwnerAvatar(owner: string | null | undefined): string | null {
  const key = owner?.trim() ?? "";
  const [url, setUrl] = useState<string | null>(() => {
    if (!key) return null;
    const entry = readCache(key);
    return entry?.kind === "url" ? entry.url : null;
  });

  useEffect(() => {
    if (!key) {
      setUrl(null);
      return;
    }
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
        if (!cancelled) setUrl(cached.url);
        return;
      }
      if (cached?.kind === "miss-permanent") {
        if (!cancelled) setUrl(null);
        return;
      }
      if (cached?.kind === "miss-transient") {
        if (!cancelled) setUrl(null);
        scheduleRetry(cached.until);
        return;
      }
      const next = await loadAvatar(key);
      if (cancelled) return;
      setUrl(next);
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
  }, [key]);

  return url;
}
