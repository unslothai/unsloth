// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ActivityIcon, CircleIcon, RefreshCwIcon } from "lucide-react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { getApiMonitor } from "../../chat/api/chat-api";
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
        className="h-full rounded-full bg-primary"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function MonitorEntry({ entry }: { entry: ApiMonitorEntry }): ReactElement {
  return (
    <article className="rounded-lg border border-border/70 bg-background p-3">
      <div className="flex min-w-0 items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex min-w-0 items-center gap-2">
            <CircleIcon
              className={cn("size-2.5 fill-current", statusTone(entry.status))}
            />
            <span className="truncate text-xs font-medium">
              {compactEndpoint(entry.endpoint)}
            </span>
          </div>
          <div className="mt-1 truncate text-[11px] text-muted-foreground">
            {entry.model}
          </div>
        </div>
        <div className="shrink-0 text-right text-[11px] text-muted-foreground">
          <div>{formatTime(entry.started_at)}</div>
          <div>{formatDuration(entry.duration_ms)}</div>
        </div>
      </div>

      <div className="mt-3 grid gap-2">
        <div>
          <div className="mb-1 text-[10px] font-semibold uppercase text-muted-foreground">
            Prompt
          </div>
          <pre className="max-h-28 overflow-auto whitespace-pre-wrap break-words rounded-md bg-muted/45 p-2 text-xs leading-5">
            {entry.prompt || "No prompt text"}
          </pre>
        </div>
        <div>
          <div className="mb-1 text-[10px] font-semibold uppercase text-muted-foreground">
            Reply
          </div>
          <pre className="max-h-28 overflow-auto whitespace-pre-wrap break-words rounded-md bg-muted/45 p-2 text-xs leading-5">
            {entry.error ||
              entry.reply ||
              (entry.status === "running" ? "Waiting..." : "No reply")}
          </pre>
        </div>
      </div>

      <div className="mt-3 text-[11px] text-muted-foreground">
        {formatTokens(entry)}
        {entry.context_length ? (
          <> / {entry.context_length.toLocaleString()} context</>
        ) : null}
        <UsageBar value={entry.context_usage} />
      </div>
    </article>
  );
}

export function ApiMonitorConsole(): ReactElement {
  const [data, setData] = useState<ApiMonitorResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

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
              <MonitorEntry key={entry.id} entry={entry} />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
