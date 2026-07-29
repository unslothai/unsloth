// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LruMap } from "./lru-map.ts";

export type AvatarCacheEntry =
  | { kind: "url"; url: string; expiresAt: number }
  | { kind: "miss-permanent" }
  | {
      kind: "miss-transient";
      until: number;
      failures: number;
      staleUrl?: string;
    };

type Listener = () => void;

export class HfOwnerAvatarCache {
  private readonly entries: LruMap<string, AvatarCacheEntry>;
  private readonly listeners = new Map<string, Set<Listener>>();

  constructor(maxEntries: number) {
    this.entries = new LruMap(maxEntries);
  }

  get(owner: string): AvatarCacheEntry | undefined {
    return this.entries.get(owner);
  }

  set(owner: string, entry: AvatarCacheEntry): void {
    const evictedOwner = this.entries.set(owner, entry);
    if (evictedOwner !== undefined && evictedOwner !== owner) {
      this.notify(evictedOwner);
    }
    this.notify(owner);
  }

  private notify(owner: string): void {
    for (const listener of this.listeners.get(owner) ?? []) {
      listener();
    }
  }

  getUrl(owner: string): string | null {
    if (!owner) {
      return null;
    }
    const entry = this.get(owner);
    if (entry?.kind === "url") {
      return entry.url;
    }
    return entry?.kind === "miss-transient" ? (entry.staleUrl ?? null) : null;
  }

  subscribe(owner: string, listener: Listener): () => void {
    const ownerListeners = this.listeners.get(owner) ?? new Set<Listener>();
    ownerListeners.add(listener);
    this.listeners.set(owner, ownerListeners);
    return () => {
      ownerListeners.delete(listener);
      if (ownerListeners.size === 0) {
        this.listeners.delete(owner);
      }
    };
  }
}

export const hfOwnerAvatarCache = new HfOwnerAvatarCache(256);
