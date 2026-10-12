// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface RowPageResult<T> {
  rows: T[];
  nextCursor: string | null;
  hasMore: boolean;
  total?: number | null;
}

export type FetchRowPage<Q, T> = (
  query: Q,
  limit: number,
  cursor: string | null,
) => Promise<RowPageResult<T>>;

export interface RowPagerState<T> {
  rows: T[];
  hasMore: boolean;
  total: number | null;
  loaded: boolean;
  loading: boolean;
  error: unknown;
}

export interface RowPagerOptions {
  initialLimit: number;
  pageSize: number;
  maxRequestLimit?: number;
}

const DEFAULT_MAX_REQUEST_LIMIT = 200;

function dedupeAppend<T extends { id: string }>(head: T[], tail: T[]): T[] {
  const seen = new Set(head.map((row) => row.id));
  const out = [...head];
  for (const row of tail) {
    if (seen.has(row.id)) continue;
    seen.add(row.id);
    out.push(row);
  }
  return out;
}

/** One sidebar list read a page at a time; a refresh re-reads only the loaded window, and answers
 *  from an older generation (query change or refresh) are dropped. */
export class RowPager<Q, T extends { id: string }> {
  private state: RowPagerState<T> = {
    rows: [],
    hasMore: false,
    total: null,
    loaded: false,
    loading: false,
    error: null,
  };
  private cursor: string | null = null;
  private generation = 0;
  private pending: Promise<void> | null = null;
  private readonly listeners = new Set<() => void>();
  private readonly maxRequestLimit: number;
  private readonly fetchPage: FetchRowPage<Q, T>;
  private query: Q;
  private queryKey: string;
  private readonly options: RowPagerOptions;

  constructor(
    fetchPage: FetchRowPage<Q, T>,
    query: Q,
    queryKey: string,
    options: RowPagerOptions,
  ) {
    this.fetchPage = fetchPage;
    this.query = query;
    this.queryKey = queryKey;
    this.options = options;
    this.maxRequestLimit = options.maxRequestLimit ?? DEFAULT_MAX_REQUEST_LIMIT;
  }

  getState(): RowPagerState<T> {
    return this.state;
  }

  getQuery(): Q {
    return this.query;
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private patch(next: Partial<RowPagerState<T>>): void {
    this.state = { ...this.state, ...next };
    for (const listener of this.listeners) listener();
  }

  private async read(
    cursor: string | null,
    count: number,
  ): Promise<{
    rows: T[];
    cursor: string | null;
    hasMore: boolean;
    total: number | null;
  }> {
    let rows: T[] = [];
    let next = cursor;
    let hasMore = true;
    let total: number | null = null;
    while (rows.length < count && hasMore) {
      const limit = Math.min(this.maxRequestLimit, count - rows.length);
      const page = await this.fetchPage(this.query, limit, next);
      rows = dedupeAppend(rows, page.rows);
      next = page.nextCursor;
      hasMore = page.hasMore && page.nextCursor !== null;
      total = page.total ?? total;
      if (page.rows.length === 0) break;
    }
    return { rows, cursor: next, hasMore, total };
  }

  start(): Promise<void> {
    if (this.state.loaded || this.state.loading)
      return this.pending ?? Promise.resolve();
    return this.refresh();
  }

  /** Keeps the loaded depth, so a re-sort or a pin does not reset the scroll to page one. */
  setQuery(query: Q, queryKey: string): Promise<void> {
    if (queryKey === this.queryKey) return Promise.resolve();
    this.query = query;
    this.queryKey = queryKey;
    if (!this.state.loaded && !this.state.loading) return Promise.resolve();
    return this.refresh();
  }

  refresh(): Promise<void> {
    const generation = ++this.generation;
    const window = Math.max(this.options.initialLimit, this.state.rows.length);
    this.patch({ loading: true });
    const run = (async () => {
      try {
        const result = await this.read(null, window);
        if (generation !== this.generation) return;
        this.cursor = result.cursor;
        this.patch({
          rows: result.rows,
          hasMore: result.hasMore,
          total: result.total,
          loaded: true,
          loading: false,
          error: null,
        });
      } catch (error) {
        if (generation !== this.generation) return;
        this.patch({ loaded: true, loading: false, error });
      }
    })();
    this.pending = run;
    return run;
  }

  private extend(count: number): Promise<void> {
    if (this.state.loading) return this.pending ?? Promise.resolve();
    if (!this.state.loaded || !this.state.hasMore || this.state.error) {
      return Promise.resolve();
    }
    const generation = this.generation;
    this.patch({ loading: true });
    const run = (async () => {
      try {
        const result = await this.read(this.cursor, count);
        if (generation !== this.generation) return;
        this.cursor = result.cursor;
        this.patch({
          rows: dedupeAppend(this.state.rows, result.rows),
          hasMore: result.hasMore,
          total: result.total ?? this.state.total,
          loading: false,
        });
      } catch (error) {
        if (generation !== this.generation) return;
        if ((error as { cursorRejected?: boolean } | null)?.cursorRejected) {
          // stale cursor: re-read the same depth from the top
          this.patch({ loading: false });
          await this.refresh();
          return;
        }
        this.patch({ loading: false, error });
      }
    })();
    this.pending = run;
    return run;
  }

  loadMore(): Promise<void> {
    return this.extend(this.options.pageSize);
  }

  async loadAll(): Promise<void> {
    await this.start();
    for (let guard = 0; guard < 10_000; guard++) {
      if (this.state.loading) {
        await this.pending;
        continue;
      }
      if (!this.state.hasMore || this.state.error) return;
      await this.extend(Number.MAX_SAFE_INTEGER);
    }
  }

  retry(): Promise<void> {
    if (!this.state.error) return Promise.resolve();
    if (this.state.rows.length === 0) return this.refresh();
    this.patch({ error: null });
    return this.loadMore();
  }
}
