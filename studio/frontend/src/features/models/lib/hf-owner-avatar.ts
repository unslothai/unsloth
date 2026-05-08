// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";

const cache = new Map<string, string | null>();
const inflight = new Map<string, Promise<string | null>>();

async function fetchAvatarUrl(name: string): Promise<string | null> {
  const candidates = [
    `https://huggingface.co/api/organizations/${encodeURIComponent(name)}/overview`,
    `https://huggingface.co/api/users/${encodeURIComponent(name)}/overview`,
  ];

  for (const url of candidates) {
    try {
      const res = await fetch(url, { credentials: "omit" });
      if (!res.ok) continue;
      const data = (await res.json()) as { avatarUrl?: string };
      if (data.avatarUrl) {
        return data.avatarUrl.startsWith("http")
          ? data.avatarUrl
          : `https://huggingface.co${data.avatarUrl}`;
      }
    } catch {
      continue;
    }
  }
  return null;
}

function loadAvatar(name: string): Promise<string | null> {
  const existing = inflight.get(name);
  if (existing) return existing;
  const promise = fetchAvatarUrl(name).then((url) => {
    cache.set(name, url);
    inflight.delete(name);
    return url;
  });
  inflight.set(name, promise);
  return promise;
}

export function useHfOwnerAvatar(owner: string | null | undefined): string | null {
  const key = owner?.trim() ?? "";
  const initial = key && cache.has(key) ? cache.get(key) ?? null : null;
  const [url, setUrl] = useState<string | null>(initial);

  useEffect(() => {
    if (!key) {
      setUrl(null);
      return;
    }
    if (cache.has(key)) {
      setUrl(cache.get(key) ?? null);
      return;
    }
    let cancelled = false;
    void loadAvatar(key).then((next) => {
      if (!cancelled) setUrl(next);
    });
    return () => {
      cancelled = true;
    };
  }, [key]);

  return url;
}
