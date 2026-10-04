// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { BrowserPage } from "./api";

const MAX_PAGES = 12;
// Total and per-page budget: a fetched page can be 50 MB.
const MAX_TOTAL_BYTES = 32 * 1024 * 1024;
const MAX_PAGE_BYTES = 8 * 1024 * 1024;

/** Approximate memory held by a page. JS strings are UTF-16. */
export function pageBytes(page: BrowserPage): number {
  return page.kind === "raw" ? page.blob.size : page.html.length * 2;
}

type Cached = { page: BrowserPage; bytes: number };

/** Loaded pages by history entry, least recently used first, within a byte budget. */
export class PageCache<Key extends object> {
  private readonly pages = new Map<Key, Cached>();
  private total = 0;
  private readonly maxPages: number;
  private readonly maxTotalBytes: number;
  private readonly maxPageBytes: number;

  constructor(maxPages = MAX_PAGES, maxTotalBytes = MAX_TOTAL_BYTES, maxPageBytes = MAX_PAGE_BYTES) {
    this.maxPages = maxPages;
    this.maxTotalBytes = maxTotalBytes;
    this.maxPageBytes = maxPageBytes;
  }

  get bytes(): number {
    return this.total;
  }

  get size(): number {
    return this.pages.size;
  }

  get(key: Key): BrowserPage | undefined {
    const hit = this.pages.get(key);
    if (!hit) return undefined;
    this.pages.delete(key);
    this.pages.set(key, hit);
    return hit.page;
  }

  set(key: Key, page: BrowserPage): void {
    this.delete(key);
    const bytes = pageBytes(page);
    if (bytes > this.maxPageBytes) return;
    this.pages.set(key, { page, bytes });
    this.total += bytes;
    for (const [oldest] of this.pages) {
      if (this.pages.size <= this.maxPages && this.total <= this.maxTotalBytes) break;
      this.delete(oldest);
    }
  }

  clear(): void {
    this.pages.clear();
    this.total = 0;
  }

  delete(key: Key): void {
    const hit = this.pages.get(key);
    if (!hit) return;
    this.pages.delete(key);
    this.total -= hit.bytes;
  }
}
