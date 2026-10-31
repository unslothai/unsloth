// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Byte-budgeted LRU of object URLs for auth-protected gallery media, which would otherwise be pinned
// all session. The caller drives recency; prune never evicts a protected id.

export interface CachedBlobUrl {
  url: string;
  bytes: number;
}

export class BlobUrlCache {
  // Insertion order IS the LRU order: touch() re-inserts.
  private readonly entries = new Map<string, CachedBlobUrl>();
  private totalBytes = 0;
  // A field, not a parameter property: erasableSyntaxOnly forbids that form.
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

  ids(): string[] {
    return [...this.entries.keys()];
  }

  toRecord(): Record<string, string> {
    const out: Record<string, string> = {};
    for (const [id, entry] of this.entries) out[id] = entry.url;
    return out;
  }

  touch(id: string): void {
    const entry = this.entries.get(id);
    if (entry === undefined) return;
    this.entries.delete(id);
    this.entries.set(id, entry);
  }

  /** Replacing an id revokes the URL it had. */
  set(id: string, url: string, bytes: number): void {
    this.delete(id);
    this.entries.set(id, { url, bytes });
    this.totalBytes += bytes;
  }

  delete(id: string): boolean {
    const entry = this.entries.get(id);
    if (entry === undefined) return false;
    this.entries.delete(id);
    this.totalBytes -= entry.bytes;
    URL.revokeObjectURL(entry.url);
    return true;
  }

  clear(): void {
    for (const entry of this.entries.values()) URL.revokeObjectURL(entry.url);
    this.entries.clear();
    this.totalBytes = 0;
  }

  /** Returns evicted ids so callers drop them from render state. */
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
