// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A byte-budgeted LRU of object URLs for auth-protected gallery media. The endpoints need an auth header, so each item is fetched into a blob
// wrapped in an object URL, which pins it until revoked. The galleries stay mounted after the first visit, so an unbounded map grows for the
// whole session and scrolling a few pages could pin gigabytes. Budgeting by BYTES matters because clip sizes vary by two orders of magnitude.
// Recency is driven by the caller (``touch``), and ``prune`` never evicts a protected id, so on-screen media stays resident.

export interface CachedBlobUrl {
  url: string;
  bytes: number;
}

export class BlobUrlCache {
  // Insertion order IS the LRU order: touch() re-inserts, so the oldest use is always first.
  private readonly entries = new Map<string, CachedBlobUrl>();
  private totalBytes = 0;
  // Declared as a field, not a constructor parameter property: tsconfig sets erasableSyntaxOnly, so that form is a build error.
  private readonly budgetBytes: number;

  constructor(budgetBytes: number) {
    this.budgetBytes = budgetBytes;
  }

  has(id: string): boolean {
    return this.entries.has(id);
  }

  get(id: string): string | undefined {
    return this.entries.get(id)?.url;
  }

  get size(): number {
    return this.entries.size;
  }

  get bytes(): number {
    return this.totalBytes;
  }

  /** All cached ids, least recently used first. */
  ids(): string[] {
    return [...this.entries.keys()];
  }

  /** ``{id: url}``, for seeding a component's render state on mount. */
  toRecord(): Record<string, string> {
    const out: Record<string, string> = {};
    for (const [id, entry] of this.entries) out[id] = entry.url;
    return out;
  }

  /** Mark ``id`` as most recently used. No-op for an id that is not cached. */
  touch(id: string): void {
    const entry = this.entries.get(id);
    if (entry === undefined) return;
    this.entries.delete(id);
    this.entries.set(id, entry);
  }

  /** Cache ``url`` for ``id``. Replacing an id revokes the URL it had. */
  set(id: string, url: string, bytes: number): void {
    this.delete(id);
    this.entries.set(id, { url, bytes });
    this.totalBytes += bytes;
  }

  /** Drop ``id`` and revoke its URL. Returns whether anything was cached. */
  delete(id: string): boolean {
    const entry = this.entries.get(id);
    if (entry === undefined) return false;
    this.entries.delete(id);
    this.totalBytes -= entry.bytes;
    URL.revokeObjectURL(entry.url);
    return true;
  }

  /** Drop and revoke everything. */
  clear(): void {
    for (const entry of this.entries.values()) URL.revokeObjectURL(entry.url);
    this.entries.clear();
    this.totalBytes = 0;
  }

  /** Evict least-recently-used entries until the total is within budget, skipping ``protectedIds``. Returns the evicted ids so the caller can drop them from render state; those cards re-fetch if they come back into view. */
  prune(protectedIds: Iterable<string> = []): string[] {
    const keep = protectedIds instanceof Set ? protectedIds : new Set(protectedIds);
    const evicted: string[] = [];
    for (const id of [...this.entries.keys()]) {
      if (this.totalBytes <= this.budgetBytes) break;
      if (keep.has(id)) continue;
      if (this.delete(id)) evicted.push(id);
    }
    return evicted;
  }
}
