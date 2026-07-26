// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  clearApiMonitor,
  getApiMonitor,
  getApiMonitorEntry,
} from "@/features/chat/api/chat-api";
import type {
  ApiMonitorEntry,
  ApiMonitorResponse,
} from "@/features/chat/types/api";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/** Poll cadence while the monitor is live. Matches the settings console it replaces. */
const POLL_INTERVAL_MS = 1500;

export type MonitorStatusFilter =
  | "all"
  | "running"
  | "completed"
  | "error"
  | "cancelled";

export interface MonitorStats {
  active: number;
  total: number;
  completed: number;
  errors: number;
  cancelled: number;
  /** Mean duration over finished requests, or null when none have finished. */
  avgDurationMs: number | null;
  /** Slowest finished request, for spotting a single pathological call. */
  maxDurationMs: number | null;
  totalTokens: number;
  /** Share of finished requests that failed, 0-1. Null when nothing finished. */
  errorRate: number | null;
  /** Mean completion tokens per second over requests that reported both. */
  tokensPerSecond: number | null;
}

function isTerminal(entry: ApiMonitorEntry): boolean {
  return entry.status !== "running";
}

function completionTokens(entry: ApiMonitorEntry): number | null {
  if (entry.completion_tokens != null) {
    return entry.completion_tokens;
  }
  // Some providers only report a total; subtracting the prompt is the best
  // available estimate of what was actually generated.
  if (entry.total_tokens != null && entry.prompt_tokens != null) {
    return Math.max(0, entry.total_tokens - entry.prompt_tokens);
  }
  return null;
}

function entryTokens(entry: ApiMonitorEntry): number {
  if (entry.total_tokens != null) {
    return entry.total_tokens;
  }
  return (entry.prompt_tokens ?? 0) + (entry.completion_tokens ?? 0);
}

export function computeStats(entries: ApiMonitorEntry[]): MonitorStats {
  let active = 0;
  let completed = 0;
  let errors = 0;
  let cancelled = 0;
  let totalTokens = 0;
  let durationSum = 0;
  let durationCount = 0;
  let maxDurationMs: number | null = null;
  // Throughput is aggregated as total tokens over total time, not as the mean of
  // each request's rate. Averaging rates lets one tiny fast request outweigh a
  // long slow one, which is the opposite of what someone debugging wants to see.
  let generatedTokens = 0;
  let generatedDurationMs = 0;

  for (const entry of entries) {
    totalTokens += entryTokens(entry);
    if (entry.status === "running") {
      active += 1;
    } else if (entry.status === "error") {
      errors += 1;
    } else if (entry.status === "cancelled") {
      cancelled += 1;
    } else {
      completed += 1;
    }
    const duration = entry.duration_ms;
    if (duration != null && isTerminal(entry)) {
      durationSum += duration;
      durationCount += 1;
      maxDurationMs =
        maxDurationMs == null ? duration : Math.max(maxDurationMs, duration);
      const generated = completionTokens(entry);
      // Sub-millisecond durations would divide into a meaningless rate.
      if (generated != null && generated > 0 && duration > 0) {
        generatedTokens += generated;
        generatedDurationMs += duration;
      }
    }
  }

  const finished = completed + errors + cancelled;
  return {
    active,
    total: entries.length,
    completed,
    errors,
    cancelled,
    avgDurationMs: durationCount > 0 ? durationSum / durationCount : null,
    maxDurationMs,
    totalTokens,
    errorRate: finished > 0 ? errors / finished : null,
    tokensPerSecond:
      generatedDurationMs > 0
        ? generatedTokens / (generatedDurationMs / 1000)
        : null,
  };
}

export function filterEntries(
  entries: ApiMonitorEntry[],
  status: MonitorStatusFilter,
  query: string,
): ApiMonitorEntry[] {
  const needle = query.trim().toLowerCase();
  return entries.filter((entry) => {
    if (status !== "all" && entry.status !== status) {
      return false;
    }
    if (!needle) {
      return true;
    }
    // Search the fields a debugging session actually keys off: which model,
    // which endpoint, and the previews/error text visible in the row.
    return (
      entry.model.toLowerCase().includes(needle) ||
      entry.endpoint.toLowerCase().includes(needle) ||
      entry.prompt_preview.toLowerCase().includes(needle) ||
      entry.reply_preview.toLowerCase().includes(needle) ||
      (entry.error ?? "").toLowerCase().includes(needle)
    );
  });
}

interface UseApiMonitorResult {
  data: ApiMonitorResponse | null;
  entries: ApiMonitorEntry[];
  stats: MonitorStats;
  error: string | null;
  /** True until the first response lands, so the page can show skeletons once. */
  loading: boolean;
  refreshing: boolean;
  paused: boolean;
  setPaused: (paused: boolean) => void;
  refresh: () => void;
  clear: () => Promise<void>;
  /** Full prompt/reply for entries the user expanded, keyed by entry id. */
  details: Record<string, ApiMonitorEntry>;
  loadingDetails: ReadonlySet<string>;
  requestDetail: (id: string) => void;
}

/**
 * Live view of the server's OpenAI-compatible API traffic.
 *
 * Polls rather than streams because the backing monitor is an in-memory ring
 * buffer with no change feed. Polling is self-rescheduling (never overlapping),
 * and pausing stops it entirely so a user reading a stalled request's payload
 * isn't fighting a list that reorders under them.
 *
 * `intervalMs` lets a caller trade freshness for cost: the full page wants the
 * default live cadence, while the floating overlay slows right down when it is
 * closed and only watching for the traffic that should pop it open.
 */
export function useApiMonitor({
  intervalMs = POLL_INTERVAL_MS,
}: { intervalMs?: number } = {}): UseApiMonitorResult {
  const [data, setData] = useState<ApiMonitorResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [paused, setPaused] = useState(false);
  const [details, setDetails] = useState<Record<string, ApiMonitorEntry>>({});
  const [loadingDetails, setLoadingDetails] = useState<Set<string>>(
    () => new Set(),
  );
  // Mirrors `loadingDetails` outside React state so the fetch guard sees writes
  // from the same tick (state updates are async and would let duplicates through).
  const inFlightDetails = useRef<Set<string>>(new Set());

  const load = useCallback(async (): Promise<void> => {
    setRefreshing(true);
    try {
      const next = await getApiMonitor();
      setData(next);
      setError(null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Monitor unavailable");
    } finally {
      setRefreshing(false);
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (paused) {
      return;
    }
    let cancelled = false;
    let timer: number | undefined;

    function poll(): void {
      getApiMonitor()
        .then((next) => {
          if (cancelled) return;
          setData(next);
          setError(null);
        })
        .catch((err: unknown) => {
          if (cancelled) return;
          setError(err instanceof Error ? err.message : "Monitor unavailable");
        })
        .finally(() => {
          if (cancelled) return;
          setLoading(false);
          timer = window.setTimeout(poll, intervalMs);
        });
    }

    poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [paused, intervalMs]);

  const requestDetail = useCallback((id: string): void => {
    if (inFlightDetails.current.has(id)) {
      return;
    }
    inFlightDetails.current.add(id);
    setLoadingDetails((prev) => new Set(prev).add(id));
    getApiMonitorEntry(id)
      .then((entry) => {
        setDetails((prev) => ({ ...prev, [id]: entry }));
      })
      .catch(() => {
        // The entry aged out of the ring buffer; drop any stale copy so the UI
        // falls back to the row previews instead of showing a frozen payload.
        setDetails((prev) => {
          if (!(id in prev)) return prev;
          const next = { ...prev };
          delete next[id];
          return next;
        });
      })
      .finally(() => {
        inFlightDetails.current.delete(id);
        setLoadingDetails((prev) => {
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
      });
  }, []);

  const clear = useCallback(async (): Promise<void> => {
    await clearApiMonitor();
    setDetails({});
    await load();
  }, [load]);

  const entries = useMemo(() => data?.entries ?? [], [data]);
  const stats = useMemo(() => computeStats(entries), [entries]);

  return {
    data,
    entries,
    stats,
    error,
    loading,
    refreshing,
    paused,
    setPaused,
    refresh: () => void load(),
    clear,
    details,
    loadingDetails,
    requestDetail,
  };
}
