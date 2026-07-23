// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  ActivityIcon,
  ChevronDownIcon,
  CircleIcon,
  RefreshCwIcon,
} from "lucide-react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { getApiMonitor, getApiMonitorEntry } from "../../chat/api/chat-api";
import type { ApiMonitorEntry, ApiMonitorResponse } from "../../chat/types/api";

const API_INFERENCE_PREFIX_RE = /^\/api\/inference/;
const V1_PREFIX_RE = /^\/v1\//;

function formatTime(value: number): string {
  return new Date(value * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatDuration(value?: number | null): string {
  if (value == null) {
    return "Running";
  }
  if (value < 1000) {
    return `${value} ms`;
  }
  return `${(value / 1000).toFixed(value < 10000 ? 1 : 0)} s`;
}

function formatTokens(entry: ApiMonitorEntry): string {
  if (entry.total_tokens != null) {
    return `${entry.total_tokens.toLocaleString()} tokens`;
  }
  if (entry.prompt_tokens != null || entry.completion_tokens != null) {
    const prompt = entry.prompt_tokens ?? 0;
    const completion = entry.completion_tokens ?? 0;
    return `${(prompt + completion).toLocaleString()} tokens`;
  }
  return "Tokens pending";
}

function compactEndpoint(endpoint: string): string {
  return endpoint
    .replace(API_INFERENCE_PREFIX_RE, "/api")
    .replace(V1_PREFIX_RE, "/");
}

function statusTone(status: ApiMonitorEntry["status"]): string {
  if (status === "running") {
    return "text-emerald-500";
  }
  if (status === "error") {
    return "text-destructive";
  }
  if (status === "cancelled") {
    return "text-amber-500";
  }
  return "text-muted-foreground";
}

function UsageBar({ value }: { value?: number | null }): ReactElement | null {
  if (value == null) {
    return null;
  }
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-muted">
      <div
        className="h-full rounded-full bg-control-accent"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function MonitorEntry({
  entry,
  detail,
  expanded,
  loading,
  onToggle,
}: {
  entry: ApiMonitorEntry;
  detail?: ApiMonitorEntry;
  expanded: boolean;
  loading: boolean;
  onToggle: () => void;
}): ReactElement {
  const hasCurrentDetail =
    detail &&
    detail.status === entry.status &&
    detail.updated_at >= entry.updated_at;
  const prompt = detail?.prompt ?? entry.prompt_preview;
  const replyText = hasCurrentDetail
    ? detail.error ?? detail.reply ?? entry.error ?? entry.reply_preview
    : entry.error ?? entry.reply_preview;
  const reply = replyText || (entry.status === "running" ? "Waiting..." : "No reply");

  return (
    <article className="min-w-0 rounded-lg border border-border/70 bg-background">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full min-w-0 items-start justify-between gap-3 p-3 text-left"
        aria-expanded={expanded}
      >
        <div className="min-w-0">
          <div className="flex min-w-0 items-center gap-2">
            <CircleIcon
              className={cn("size-2.5 fill-current", statusTone(entry.status))}
            />
            <span className="truncate text-xs font-medium">
              {compactEndpoint(entry.endpoint)}
            </span>
          </div>
          <div className="mt-1 truncate text-ui-11 text-muted-foreground">
            {entry.model}
          </div>
          <div className="mt-2 line-clamp-2 whitespace-pre-wrap break-words text-xs text-muted-foreground">
            {entry.error ||
              entry.reply_preview ||
              entry.prompt_preview ||
              (entry.status === "running" ? "Waiting..." : "No preview")}
          </div>
        </div>
        <div className="flex shrink-0 items-start gap-2 text-right text-ui-11 text-muted-foreground">
          <div>
            <div>{formatTime(entry.started_at)}</div>
            <div>{formatDuration(entry.duration_ms)}</div>
          </div>
          <ChevronDownIcon
            className={cn(
              "mt-0.5 size-3.5 transition-transform",
              expanded && "rotate-180",
            )}
          />
        </div>
      </button>

      {expanded ? (
        <div className="border-t border-border/60 p-3 pt-2">
          <div className="grid gap-2">
            <div>
              <div className="mb-1 flex items-center justify-between gap-2 text-ui-10 font-semibold uppercase text-muted-foreground">
                <span>Prompt</span>
                {entry.prompt_truncated && !detail ? <span>Preview</span> : null}
              </div>
              <pre className="max-h-44 overflow-auto whitespace-pre-wrap break-words rounded-md bg-muted/45 p-2 text-xs leading-5">
                {loading && !detail ? "Loading..." : prompt || "No prompt text"}
              </pre>
            </div>
            <div>
              <div className="mb-1 flex items-center justify-between gap-2 text-ui-10 font-semibold uppercase text-muted-foreground">
                <span>Reply</span>
                {entry.reply_truncated && !detail ? <span>Preview</span> : null}
              </div>
              <pre className="max-h-44 overflow-auto whitespace-pre-wrap break-words rounded-md bg-muted/45 p-2 text-xs leading-5">
                {loading && !detail ? "Loading..." : reply}
              </pre>
            </div>
          </div>

          <div className="mt-3 text-ui-11 text-muted-foreground">
            {formatTokens(entry)}
            {entry.context_length ? (
              <> / {entry.context_length.toLocaleString()} context</>
            ) : null}
            <UsageBar value={entry.context_usage} />
          </div>
        </div>
      ) : null}
    </article>
  );
}

export function ApiMonitorConsole(): ReactElement {
  const [data, setData] = useState<ApiMonitorResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(() => new Set());
  const [details, setDetails] = useState<Record<string, ApiMonitorEntry>>({});
  const [loadingDetails, setLoadingDetails] = useState<Set<string>>(
    () => new Set(),
  );
  const loadingDetailsRef = useRef<Set<string>>(new Set());
  const detailsRef = useRef<Record<string, ApiMonitorEntry>>({});

  const loadMonitor = useCallback(async (): Promise<void> => {
    setRefreshing(true);
    try {
      setData(await getApiMonitor());
      setError(null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Monitor unavailable");
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    function schedule(): void {
      timer = window.setTimeout(poll, 1500);
    }

    function poll(): void {
      getApiMonitor()
        .then((next) => {
          if (cancelled) {
            return;
          }
          setData(next);
          setError(null);
        })
        .catch((err: unknown) => {
          if (cancelled) {
            return;
          }
          setError(err instanceof Error ? err.message : "Monitor unavailable");
        })
        .finally(() => {
          if (!cancelled) {
            schedule();
          }
        });
    }

    poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, []);

  const statusLabel = data?.status ?? "idle";
  const hasActive = (data?.active_requests ?? 0) > 0;
  const entries = useMemo(() => data?.entries ?? [], [data]);
  const loadDetail = useCallback(
    (id: string): void => {
      if (loadingDetailsRef.current.has(id)) {
        return;
      }
      loadingDetailsRef.current.add(id);
      setLoadingDetails((prev) => new Set(prev).add(id));
      getApiMonitorEntry(id)
        .then((entry) => {
          setDetails((prev) => {
            const next = { ...prev, [id]: entry };
            detailsRef.current = next;
            return next;
          });
        })
        .catch(() => {
          setDetails((prev) => {
            const next = { ...prev };
            delete next[id];
            detailsRef.current = next;
            return next;
          });
        })
        .finally(() => {
          loadingDetailsRef.current.delete(id);
          setLoadingDetails((prev) => {
            const next = new Set(prev);
            next.delete(id);
            return next;
          });
        });
    },
    [],
  );

  const toggleEntry = useCallback(
    (entry: ApiMonitorEntry): void => {
      setExpandedIds((prev) => {
        const next = new Set(prev);
        if (next.has(entry.id)) {
          next.delete(entry.id);
        } else {
          next.add(entry.id);
          loadDetail(entry.id);
        }
        return next;
      });
    },
    [loadDetail],
  );

  useEffect(() => {
    for (const entry of entries) {
      if (!expandedIds.has(entry.id)) {
        continue;
      }
      const cached = detailsRef.current[entry.id];
      if (!cached || cached.status !== entry.status || entry.status === "running") {
        loadDetail(entry.id);
      }
    }
  }, [entries, expandedIds, loadDetail]);

  return (
    <section className="flex min-w-0 flex-col rounded-lg border border-border/70 bg-background">
      <div className="flex min-w-0 items-start justify-between gap-3 border-b border-border/60 px-4 py-3">
        <div className="flex min-w-0 gap-3">
          <div className="relative mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-md border border-border/70 bg-muted/40">
            <ActivityIcon className="size-4 text-foreground" />
            {hasActive ? (
              <span className="absolute right-1 top-1 size-2 rounded-full bg-emerald-500" />
            ) : null}
          </div>
          <div className="min-w-0">
            <h2 className="text-sm font-semibold text-foreground">
              API monitor
            </h2>
            <p className="truncate text-xs text-muted-foreground">
              {data?.active_model ?? "No model loaded"}
            </p>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <div className="rounded-full border border-border px-2.5 py-1 text-xs capitalize text-muted-foreground">
            {statusLabel}
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => void loadMonitor()}
            disabled={refreshing}
          >
            <RefreshCwIcon
              className={cn("size-3.5", refreshing && "animate-spin")}
            />
            Refresh
          </Button>
        </div>
      </div>

      <div className="flex items-center justify-between border-b border-border/60 px-4 py-2 text-xs text-muted-foreground">
        <span>
          {(data?.active_requests ?? 0).toLocaleString()} active /{" "}
          {entries.length.toLocaleString()} recent
        </span>
        {data?.context_length ? (
          <span>{data.context_length.toLocaleString()} context</span>
        ) : null}
      </div>

      <div className="max-h-[420px] min-h-24 overflow-y-auto p-3">
        {error ? (
          <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        ) : entries.length === 0 ? (
          <div className="rounded-lg border border-border/70 p-4 text-sm text-muted-foreground">
            No API traffic yet
          </div>
        ) : (
          <div className="grid gap-3">
            {entries.map((entry) => (
              <MonitorEntry
                key={entry.id}
                entry={entry}
                detail={details[entry.id]}
                expanded={expandedIds.has(entry.id)}
                loading={loadingDetails.has(entry.id)}
                onToggle={() => toggleEntry(entry)}
              />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
