// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Tooltip, TooltipContent } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { ActivityIcon, CircleIcon, RefreshCwIcon } from "lucide-react";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { type ReactElement, useEffect, useMemo, useState } from "react";
import { getApiMonitor } from "../api/chat-api";
import type { ApiMonitorEntry, ApiMonitorResponse } from "../types/api";

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

export function ApiMonitorPanel(): ReactElement {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<ApiMonitorResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    function schedule(): void {
      timer = window.setTimeout(poll, open ? 1500 : 5000);
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
  }, [open]);

  const statusLabel = data?.status ?? "idle";
  const hasActive = (data?.active_requests ?? 0) > 0;
  const entries = useMemo(() => data?.entries ?? [], [data]);

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <Tooltip>
        <TooltipPrimitive.Trigger asChild={true}>
          <button
            type="button"
            onClick={() => setOpen(true)}
            className="relative flex h-[34px] w-[34px] items-center justify-center rounded-[12px] text-nav-fg transition-colors hover:bg-nav-surface-hover hover:text-black dark:hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label="Open API monitor"
          >
            <ActivityIcon className="size-4" />
            {hasActive ? (
              <span className="absolute right-1.5 top-1.5 size-2 rounded-full bg-emerald-500" />
            ) : null}
          </button>
        </TooltipPrimitive.Trigger>
        <TooltipContent
          side="bottom"
          sideOffset={6}
          className="tooltip-compact"
        >
          API monitor
        </TooltipContent>
      </Tooltip>

      <SheetContent className="w-[92vw] sm:max-w-[560px]">
        <SheetHeader className="border-b border-border/60 px-5 py-4">
          <div className="flex items-start justify-between gap-3 pr-8">
            <div className="min-w-0">
              <SheetTitle>API monitor</SheetTitle>
              <SheetDescription className="truncate">
                {data?.active_model ?? "No model loaded"}
              </SheetDescription>
            </div>
            <div className="shrink-0 rounded-full border border-border px-2.5 py-1 text-xs capitalize text-muted-foreground">
              {statusLabel}
            </div>
          </div>
        </SheetHeader>

        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
          <div className="flex items-center justify-between border-b border-border/60 px-5 py-3 text-xs text-muted-foreground">
            <span>
              {(data?.active_requests ?? 0).toLocaleString()} active /{" "}
              {entries.length.toLocaleString()} recent
            </span>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => {
                getApiMonitor()
                  .then(setData)
                  .catch(() => setError("Monitor unavailable"));
              }}
            >
              <RefreshCwIcon className="size-3.5" />
              Refresh
            </Button>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
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
        </div>
      </SheetContent>
    </Sheet>
  );
}
