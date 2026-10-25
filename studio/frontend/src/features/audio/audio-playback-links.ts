// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Under the server's one-hour link lifetime; only a new selection renews, never a playing source.
const LINK_REFRESH_MS = 50 * 60 * 1000;

export class AudioPlaybackLinks {
  private readonly entries = new Map<string, { url: string; mintedAt: number }>();
  private readonly retried = new Set<string>();

  get(id: string): string | undefined {
    return this.entries.get(id)?.url;
  }

  set(id: string, url: string, now = Date.now()): void {
    this.entries.set(id, { url, mintedAt: now });
  }

  isFresh(id: string, now = Date.now()): boolean {
    const entry = this.entries.get(id);
    return entry !== undefined && now - entry.mintedAt < LINK_REFRESH_MS;
  }

  /** Ignore old-source errors and bound retries; a recovered link may itself expire later. */
  retry(id: string, failedUrl: string, now = Date.now()): boolean {
    if (
      this.get(id) !== failedUrl ||
      (this.retried.has(id) && this.isFresh(id, now))
    ) return false;
    this.retried.add(id);
    this.entries.delete(id);
    return true;
  }

  delete(id: string): void {
    this.entries.delete(id);
    this.retried.delete(id);
  }

  clear(): void {
    this.entries.clear();
    this.retried.clear();
  }

  toRecord(): Record<string, string> {
    return Object.fromEntries(
      [...this.entries].map(([id, entry]) => [id, entry.url]),
    );
  }
}
