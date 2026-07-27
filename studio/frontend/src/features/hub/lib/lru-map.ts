// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Insertion-ordered map with a maximum size. Reads and writes move a key to
 * the most-recent position; once over capacity the least-recently-used key is
 * evicted. Bounds memory for module-level caches that live for the tab's
 * lifetime (HF READMEs, dataset sizes, owner avatars).
 */
export class LruMap<K, V> {
  private readonly map = new Map<K, V>();
  private readonly max: number;

  constructor(max: number) {
    this.max = max;
  }

  has(key: K): boolean {
    return this.map.has(key);
  }

  get(key: K): V | undefined {
    if (!this.map.has(key)) return undefined;
    const value = this.map.get(key) as V;
    this.map.delete(key);
    this.map.set(key, value);
    return value;
  }

  set(key: K, value: V): void {
    this.map.delete(key);
    this.map.set(key, value);
    if (this.map.size > this.max) {
      const oldest = this.map.keys().next().value;
      if (oldest !== undefined) this.map.delete(oldest);
    }
  }

  delete(key: K): void {
    this.map.delete(key);
  }

  clear(): void {
    this.map.clear();
  }
}

/**
 * Trim an insertion-ordered Set to `max` entries, evicting oldest-first but
 * never an entry `isProtected` reports as still live. Protected entries are
 * re-inserted at the back; `protectedScans` caps the walk at one full pass so
 * a fully-protected set over capacity exits instead of looping forever.
 */
export function evictOldestUnprotected(
  set: Set<string>,
  max: number,
  isProtected: (key: string) => boolean,
): void {
  let protectedScans = 0;
  while (set.size > max && protectedScans < set.size) {
    const oldest = set.values().next().value;
    if (oldest === undefined) break;
    if (isProtected(oldest)) {
      set.delete(oldest);
      set.add(oldest);
      protectedScans += 1;
      continue;
    }
    set.delete(oldest);
    protectedScans = 0;
  }
}
