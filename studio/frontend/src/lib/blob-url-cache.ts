// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A byte-budgeted LRU of object URLs for auth-protected gallery media.
//
// Gallery images and clips can't be plain <img>/<video> src attributes (the endpoints need the
// auth header), so each one is fetched into a blob and wrapped in an object URL. An object URL
// pins its blob until it is revoked, and the galleries keep their page component mounted after
// the first visit, so an unbounded map of them grows for the whole session: a video clip is tens
// to hundreds of MB, and scrolling a few pages of the strip could pin gigabytes in the webview.
//
// Budgeting by BYTES rather than by entry count is the point: clip sizes vary by two orders of
// magnitude, so any entry cap is either useless for long clips or wasteful for short ones.
//
// Recency is driven by the caller (``touch`` on the visibility signal that already exists for
// near-viewport fetching), and ``prune`` never evicts an id the caller marks as protected -- the
// on-screen and selected media stay resident no matter how small the budget is.

export interface CachedBlobUrl {
  url: string;
  bytes: number;
}

export class BlobUrlCache {
  // Insertion order IS the LRU order: touch() re-inserts, so the oldest use is always first.
  private readonly entries = new Map<string, CachedBlobUrl>();
  private totalBytes = 0;

  constructor(private readonly budgetBytes: number) {}

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

  /**
   * Evict least-recently-used entries until the total is within budget, skipping ``protectedIds``.
   * Returns the evicted ids so the caller can drop them from its render state; those cards then
   * re-fetch if they come back into view.
   */
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
