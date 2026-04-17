// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import type { TrainingRunSummary } from "@/features/training";
import { deleteTrainingRun, listTrainingRuns } from "@/features/training";
import { formatDuration } from "@/features/studio/sections/progress-section-lib";
import { cn } from "@/lib/utils";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useCallback, useEffect, useRef, useState } from "react";
import { Spinner } from "@/components/ui/spinner";

const PAGE_SIZE = 12;
const RUNNING_POLL_INTERVAL_MS = 5000;

const statusBadge: Record<
  string,
  { label: string; className: string }
> = {
  completed: {
    label: "Completed",
    className:
      "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-400",
  },
  stopped: {
    label: "Stopped",
    className:
      "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-400",
  },
  error: {
    label: "Error",
    className: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-400",
  },
  running: {
    label: "Running",
    className:
      "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-400",
  },
};

function catmullRomPath(points: { x: number; y: number }[]): string {
  if (points.length < 2) return "";
  const d = [`M${points[0]!.x.toFixed(1)},${points[0]!.y.toFixed(1)}`];
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[Math.max(i - 1, 0)]!;
    const p1 = points[i]!;
    const p2 = points[i + 1]!;
    const p3 = points[Math.min(i + 2, points.length - 1)]!;
    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;
    d.push(
      `C${cp1x.toFixed(1)},${cp1y.toFixed(1)} ${cp2x.toFixed(1)},${cp2y.toFixed(1)} ${p2.x.toFixed(1)},${p2.y.toFixed(1)}`,
    );
  }
  return d.join(" ");
}

function Sparkline({ values, id }: { values: number[]; id: string }): ReactElement | null {
  if (!values || values.length < 2) return null;
  let min = values[0]!;
  let max = values[0]!;
  for (let i = 1; i < values.length; i++) {
    if (values[i]! < min) min = values[i]!;
    if (values[i]! > max) max = values[i]!;
  }
  const range = max - min || 1;
  const pad = 1.5; // half stroke-width so peaks aren't clipped
  const h = 32;
  const w = 120;
  const gradientId = `sparkFill-${id}`;

  // Build points with vertical padding so the stroke isn't clipped
  const pts = values.map((v, i) => ({
    x: (i / (values.length - 1)) * w,
    y: pad + (1 - (v - min) / range) * (h - pad * 2),
  }));

  const linePath = catmullRomPath(pts);
  const last = pts[pts.length - 1]!;
  const first = pts[0]!;
  const fillPath = `${linePath} L${last.x.toFixed(1)},${h} L${first.x.toFixed(1)},${h} Z`;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-8 w-full" preserveAspectRatio="none" role="img" aria-label="Loss trend sparkline">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0.12" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path
        d={fillPath}
        fill={`url(#${gradientId})`}
        className="text-emerald-500"
      />
      <path
        d={linePath}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-emerald-500"
      />
    </svg>
  );
}

function formatRelativeTime(isoDate: string): string {
  const diff = Date.now() - new Date(isoDate).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}


interface HistoryCardGridProps {
  onSelectRun: (runId: string) => void;
}

export function HistoryCardGrid({
  onSelectRun,
}: HistoryCardGridProps): ReactElement {
  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [manualFetchInFlight, setManualFetchInFlight] = useState(false);

  const userControllerRef = useRef<AbortController | null>(null);
  const pollControllerRef = useRef<AbortController | null>(null);
  const fetchIdRef = useRef(0);
  const pollIdRef = useRef(0);

  const fetchRuns = useCallback(async (offset = 0, append = false, limit = PAGE_SIZE) => {
    // Cancel any in-flight poll so its stale response can't clobber this fresher fetch
    pollControllerRef.current?.abort();
    userControllerRef.current?.abort();
    const controller = new AbortController();
    userControllerRef.current = controller;
    const id = ++fetchIdRef.current;

    setManualFetchInFlight(true);
    setLoading(true);
    setError(null);
    try {
      const result = await listTrainingRuns(limit, offset, controller.signal);
      if (fetchIdRef.current !== id) return;
      setRuns((prev) => (append ? [...prev, ...result.runs] : result.runs));
      setTotal(result.total);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      if (fetchIdRef.current !== id) return;
      if (!append) setError("Failed to load training runs");
    } finally {
      if (fetchIdRef.current === id) {
        setLoading(false);
        setManualFetchInFlight(false);
      }
    }
  }, []);

  useEffect(() => {
    void fetchRuns(0);
    return () => {
      userControllerRef.current?.abort();
    };
  }, [fetchRuns]);

  // Poll while any run is still "running" so the card shows live progress
  const hasRunningRun = runs.some((r) => r.status === "running");
  const visibleCount = runs.length;
  useEffect(() => {
    if (!hasRunningRun) return;
    const timer = setInterval(async () => {
      if (manualFetchInFlight) return;
      pollControllerRef.current?.abort();
      const controller = new AbortController();
      pollControllerRef.current = controller;
      const pid = ++pollIdRef.current;
      try {
        const limit = Math.max(PAGE_SIZE, visibleCount);
        const result = await listTrainingRuns(limit, 0, controller.signal);
        if (pollIdRef.current !== pid) return; // stale poll — discard
        setRuns(result.runs);
        setTotal(result.total);
      } catch {
        // silently handle — poll will retry
      }
    }, RUNNING_POLL_INTERVAL_MS);
    return () => {
      clearInterval(timer);
      pollControllerRef.current?.abort();
    };
  }, [hasRunningRun, visibleCount, manualFetchInFlight]);

  const handleDelete = async () => {
    if (!deleteTarget) return;
    setDeleteError(null);
    try {
      await deleteTrainingRun(deleteTarget);
      // Optimistically remove the card so it disappears immediately
      setRuns((prev) => prev.filter((r) => r.id !== deleteTarget));
      setTotal((prev) => Math.max(0, prev - 1));
      // Re-fetch preserving visible count so offsets stay consistent for "Load more"
      const currentCount = runs.length - 1;
      const limit = Math.max(PAGE_SIZE, currentCount);
      fetchRuns(0, false, limit).catch(() => {
        // Refresh failed — card is already removed, no stale display
      });
    } catch {
      setDeleteError("Failed to delete training run. Please try again.");
    }
    setDeleteTarget(null);
  };

  if (!loading && error && runs.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-16 text-center">
        <p className="text-sm text-destructive">{error}</p>
        <Button variant="outline" size="sm" onClick={() => void fetchRuns(0)}>
          Retry
        </Button>
      </div>
    );
  }

  if (!loading && runs.length === 0) {
    return (
      <div className="flex flex-col items-center gap-2 py-16 text-center">
        <p className="text-sm text-muted-foreground">
          No training runs yet. Start your first training run in the Configure
          tab.
        </p>
      </div>
    );
  }

  return (
    <>
      {deleteError && (
        <div className="mb-4 rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-2 text-sm text-destructive">
          {deleteError}
        </div>
      )}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {runs.map((run) => {
          const badge = statusBadge[run.status] ?? statusBadge.error;
          const isRunning = run.status === "running";
          return (
            <div
              role="button"
              tabIndex={0}
              key={run.id}
              className={cn(
                "group relative flex cursor-pointer flex-col gap-3 rounded-xl border bg-card p-4 text-left transition-colors hover:border-border hover:bg-accent/30",
                isRunning
                  ? "border-blue-400/50 dark:border-blue-500/30"
                  : "border-border/60",
              )}
              onClick={() => onSelectRun(run.id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onSelectRun(run.id);
                }
              }}
            >
              <div className="flex items-center justify-between pr-6">
                <span
                  className={cn(
                    "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-semibold",
                    badge.className,
                  )}
                >
                  {isRunning && <Spinner className="size-2.5" />}
                  {badge.label}
                </span>
                <span className="text-[10px] text-muted-foreground">
                  {formatRelativeTime(run.started_at)}
                </span>
              </div>
              <div className="min-w-0">
                <p
                  className="truncate text-sm font-medium"
                  title={run.model_name}
                >
                  {run.model_name}
                </p>
                <p className="truncate text-xs text-muted-foreground">
                  {run.dataset_name}
                </p>
              </div>
              {run.loss_sparkline && run.loss_sparkline.length >= 2 && (
                <Sparkline values={run.loss_sparkline} id={run.id} />
              )}
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
                <span>
                  Loss:{" "}
                  {run.final_loss != null ? run.final_loss.toFixed(4) : "--"}
                </span>
                <span>
                  Steps: {run.final_step ?? 0}/{run.total_steps ?? "--"}
                </span>
                <span>{formatDuration(run.duration_seconds)}</span>
              </div>
              {!isRunning && (
                <button
                  type="button"
                  className="absolute right-3 top-3 rounded-md p-1 text-muted-foreground/50 opacity-0 transition-opacity hover:bg-destructive/10 hover:text-destructive group-hover:opacity-100 focus-visible:opacity-100"
                  aria-label="Delete run"
                  onClick={(e) => {
                    e.stopPropagation();
                    setDeleteTarget(run.id);
                  }}
                >
                  <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                </button>
              )}
            </div>
          );
        })}
      </div>
      {runs.length < total && (
        <div className="mt-4 flex justify-center">
          <Button
            variant="outline"
            size="sm"
            onClick={() => void fetchRuns(runs.length, true)}
            disabled={loading}
          >
            {loading ? "Loading..." : "Load more"}
          </Button>
        </div>
      )}
      {loading && runs.length === 0 && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <div
              key={`skeleton-${i}`}
              className="h-40 animate-pulse rounded-xl border bg-muted/30"
            />
          ))}
        </div>
      )}
      <AlertDialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete training run?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete this training run and all its metrics.
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => void handleDelete()}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
