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
import {
  deleteTrainingRun,
  getTrainingRunDisplayTitle,
  getTrainingRunModelSubtitle,
  emitTrainingRunDeleted,
  listTrainingRuns,
  onTrainingRunDeleted,
  onTrainingRunsChanged,
  onTrainingRunUpdated,
  useTrainingActions,
  useTrainingRuntimeStore,
} from "@/features/training";
import { formatDuration } from "@/features/studio/sections/progress-section-lib";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { cn } from "@/lib/utils";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useCallback, useEffect, useRef, useState } from "react";
import { Spinner } from "@/components/ui/spinner";
import { translate, useT } from "@/i18n";

type StudioT = ReturnType<typeof useT>;

const PAGE_SIZE = 12;
const RUNNING_POLL_INTERVAL_MS = 5000;

const statusBadge: Record<
  string,
  { className: string }
> = {
  completed: {
    className:
      "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-400",
  },
  stopped: {
    className:
      "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-400",
  },
  error: {
    className: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-400",
  },
  running: {
    className:
      "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-400",
  },
  resumed_later: {
    className:
      "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-400",
  },
};

function formatStatusLabel(status: string, t: StudioT): string {
  if (status === "completed") return t("studio.history.status.completed");
  if (status === "stopped") return t("studio.history.status.stopped");
  if (status === "running") return t("studio.history.status.running");
  if (status === "resumed_later") return t("studio.history.status.continued");
  return t("studio.history.status.error");
}

function wasContinuedInVisibleRuns(
  run: TrainingRunSummary,
  runs: TrainingRunSummary[],
): boolean {
  if ((run.status !== "stopped" && run.status !== "error") || !run.output_dir)
    return false;
  const startedAt = new Date(run.started_at).getTime();
  return runs.some(
    (other) =>
      other.id !== run.id &&
      other.output_dir === run.output_dir &&
      (other.status === "stopped" ||
        other.status === "completed" ||
        other.status === "error" ||
        other.status === "running") &&
      new Date(other.started_at).getTime() > startedAt,
  );
}

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

function Sparkline({
  values,
  id,
  ariaLabel,
}: {
  values: number[];
  id: string;
  ariaLabel: string;
}): ReactElement | null {
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

  // Vertical padding so the stroke isn't clipped.
  const pts = values.map((v, i) => ({
    x: (i / (values.length - 1)) * w,
    y: pad + (1 - (v - min) / range) * (h - pad * 2),
  }));

  const linePath = catmullRomPath(pts);
  const last = pts[pts.length - 1]!;
  const first = pts[0]!;
  const fillPath = `${linePath} L${last.x.toFixed(1)},${h} L${first.x.toFixed(1)},${h} Z`;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="h-8 w-full" preserveAspectRatio="none" role="img" aria-label={ariaLabel}>
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

function formatRelativeTime(isoDate: string, t: StudioT): string {
  const diff = Date.now() - new Date(isoDate).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return t("studio.history.relativeJustNow");
  if (mins < 60) return t("studio.history.relativeMinutesAgo", { count: mins });
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return t("studio.history.relativeHoursAgo", { count: hrs });
  const days = Math.floor(hrs / 24);
  return t("studio.history.relativeDaysAgo", { count: days });
}


interface HistoryCardGridProps {
  onSelectRun: (runId: string) => void;
  onResumeStarted?: () => void;
}

export function HistoryCardGrid({
  onSelectRun,
  onResumeStarted,
}: HistoryCardGridProps): ReactElement {
  const t = useT();
  const [runs, setRuns] = useState<TrainingRunSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [resumeTarget, setResumeTarget] = useState<string | null>(null);
  const [manualFetchInFlight, setManualFetchInFlight] = useState(false);
  const { resumeTrainingRunFromHistory } = useTrainingActions();
  const isStarting = useTrainingRuntimeStore((state) => state.isStarting);
  // Copy-link base: Cloudflare tunnel > LAN host:port > origin. The tunnel
  // registers shortly after startup, so poll (bounded) until it shows.
  const cloudflareUrl = usePlatformStore((s) => s.cloudflareUrl);
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  useEffect(() => {
    if (cloudflareUrl) return;
    let cancelled = false;
    void (async () => {
      for (let attempt = 0; attempt < 12 && !cancelled; attempt++) {
        try {
          await fetchDeviceType({ force: true });
        } catch {
          // Ignore startup blips; copy-link falls back to serverUrl/origin.
        }
        if (cancelled || usePlatformStore.getState().cloudflareUrl) return;
        await new Promise((r) => setTimeout(r, 2500));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [cloudflareUrl]);

  const userControllerRef = useRef<AbortController | null>(null);
  const pollControllerRef = useRef<AbortController | null>(null);
  const fetchIdRef = useRef(0);
  const pollIdRef = useRef(0);
  const runsLengthRef = useRef(0);

  useEffect(() => {
    runsLengthRef.current = runs.length;
  }, [runs.length]);

  const fetchRuns = useCallback(async (offset = 0, append = false, limit = PAGE_SIZE) => {
    // Cancel any in-flight poll so its stale response can't clobber this fetch.
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
      if (!append) setError(translate("studio.history.loadError"));
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

  useEffect(() => {
    const offUpdated = onTrainingRunUpdated((updated) => {
      setRuns((prev) =>
        prev.map((run) => (run.id === updated.id ? updated : run)),
      );
    });
    const offDeleted = onTrainingRunDeleted((runId) => {
      setRuns((prev) => prev.filter((run) => run.id !== runId));
      setTotal((prev) => Math.max(0, prev - 1));
    });
    const offChanged = onTrainingRunsChanged(() => {
      const limit = Math.max(PAGE_SIZE, runsLengthRef.current);
      void fetchRuns(0, false, limit);
    });
    return () => {
      offUpdated();
      offDeleted();
      offChanged();
    };
  }, [fetchRuns]);

  // Poll while any run is "running" so cards show live progress.
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
        if (pollIdRef.current !== pid) return; // stale poll
        setRuns(result.runs);
        setTotal(result.total);
      } catch {
        // ignore; poll will retry
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
      emitTrainingRunDeleted(deleteTarget);
      // Re-fetch preserving visible count so "Load more" offsets stay consistent.
      const currentCount = runs.length - 1;
      const limit = Math.max(PAGE_SIZE, currentCount);
      fetchRuns(0, false, limit).catch(() => {
        // Refresh failed; card is already removed, no stale display.
      });
    } catch {
      setDeleteError(translate("studio.history.deleteError"));
    }
    setDeleteTarget(null);
  };

  const handleResume = async (runId: string) => {
    setResumeTarget(runId);
    try {
      const ok = await resumeTrainingRunFromHistory(runId);
      if (ok) {
        onResumeStarted?.();
      }
    } finally {
      setResumeTarget(null);
    }
  };

  if (!loading && error && runs.length === 0) {
    return (
      <div
        className="flex flex-col items-center gap-2 py-16 text-center"
        aria-label={t("studio.history.title")}
      >
        <p className="text-sm text-destructive">{error}</p>
        <Button variant="outline" size="sm" onClick={() => void fetchRuns(0)}>
          {t("studio.history.retry")}
        </Button>
      </div>
    );
  }

  if (!loading && runs.length === 0) {
    return (
      <div
        className="flex flex-col items-center gap-2 py-16 text-center"
        aria-label={t("studio.history.title")}
      >
        <p className="text-sm text-muted-foreground">
          {t("studio.history.emptyDescription")}
        </p>
      </div>
    );
  }

  return (
    <div className="contents" aria-label={t("studio.history.title")}>
      {deleteError && (
        <div className="mb-4 rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-2 text-sm text-destructive">
          {deleteError}
        </div>
      )}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {runs.map((run) => {
          const wasContinued =
            run.resumed_later || wasContinuedInVisibleRuns(run, runs);
          const badge = wasContinued
            ? statusBadge.resumed_later
            : (statusBadge[run.status] ?? statusBadge.error);
          const isRunning = run.status === "running";
          const canResume = run.can_resume && !wasContinued;
          const isResuming = resumeTarget === run.id;

          const title = getTrainingRunDisplayTitle(run);
          const modelSubtitle = getTrainingRunModelSubtitle(run);

          const projectSubtitle =
            run.project_name && title !== run.project_name ? run.project_name : null;
          // Backend /p ref + its capability token. Both are required: the link
          // is useless (404s) without the signature, so don't offer to copy it.
          const canCopyPreview = !!run.preview_ref && !!run.preview_sig;
          return (
            <div
              role="button"
              tabIndex={0}
              key={run.id}
              className={cn(
                "group relative flex h-[11.5rem] cursor-pointer flex-col gap-3 rounded-xl border bg-card p-4 text-left transition-colors hover:border-border hover:bg-accent/30",
                isRunning
                  ? "border-blue-400/50 dark:border-blue-500/30"
                  : "border-border/60",
                (canResume || canCopyPreview) && "gap-2",
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
                    "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-ui-10 font-semibold",
                    badge.className,
                  )}
                >
                  {isRunning && <Spinner className="size-2.5" />}
                  {formatStatusLabel(wasContinued ? "resumed_later" : run.status, t)}
                </span>
                <span className="text-ui-10 text-muted-foreground">
                  {formatRelativeTime(run.started_at, t)}
                </span>
              </div>
              {canResume && (
                <Button
                  type="button"
                  size="xs"
                  variant="outline"
                  className="absolute bottom-3 left-4 h-6 rounded-full px-2.5 text-ui-11 leading-none shadow-sm"
                  disabled={isStarting || isResuming}
                  onClick={(e) => {
                    e.stopPropagation();
                    void handleResume(run.id);
                  }}
                >
                  {isResuming ? t("studio.history.resuming") : t("studio.history.resumeTraining")}
                </Button>
              )}
              {canCopyPreview && (
                <Button
                  type="button"
                  size="xs"
                  variant="outline"
                  className="absolute bottom-3 right-4 h-6 rounded-full px-2.5 text-ui-11 leading-none shadow-sm"
                  onClick={async (e) => {
                    e.stopPropagation();
                    // Encode each segment but keep "/" so the /p route matches.
                    const ref = (run.preview_ref ?? "")
                      .split("/")
                      .map(encodeURIComponent)
                      .join("/");
                    const base = (
                      cloudflareUrl ??
                      serverUrl ??
                      window.location.origin
                    ).replace(/\/+$/, "");
                    // The signature is a bearer capability carried as ?k=; the
                    // recipient's page forwards it on its chat requests.
                    const url = `${base}/p/${ref}?k=${encodeURIComponent(run.preview_sig ?? "")}`;
                    const ok = await copyToClipboard(url);
                    toast[ok ? "success" : "error"](
                      t(
                        ok
                          ? "studio.history.previewLinkCopied"
                          : "studio.history.previewLinkCopyFailed",
                      ),
                    );
                  }}
                >
                  {t("studio.history.copyPreviewLink")}
                </Button>
              )}
              <div className="min-w-0">
                <p
                  className="truncate text-sm font-medium"
                  title={title}
                >
                  {title}
                </p>
                {modelSubtitle && (
                  <p
                    className="truncate text-xs text-muted-foreground"
                    title={modelSubtitle}
                  >
                    {modelSubtitle}
                  </p>
                )}
                <p
                  className="truncate text-xs text-muted-foreground"
                  title={run.dataset_name}
                >
                  {run.dataset_name}
                </p>
                {projectSubtitle && (
                  <p
                    className="truncate text-xs text-muted-foreground/80"
                    title={projectSubtitle}
                  >
                    {projectSubtitle}
                  </p>
                )}
              </div>
              {run.loss_sparkline && run.loss_sparkline.length >= 2 && (
                <div className={cn((canResume || canCopyPreview) && "h-7 overflow-hidden")}>
                  <Sparkline
                    values={run.loss_sparkline}
                    id={run.id}
                    ariaLabel={t("studio.history.lossTrendSparkline")}
                  />
                </div>
              )}
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-ui-11 text-muted-foreground">
                <span>
                  {t("studio.history.loss")}:{" "}
                  {run.final_loss != null ? run.final_loss.toFixed(4) : "--"}
                </span>
                <span>
                  {t("studio.history.steps")}: {run.final_step ?? 0}/{run.total_steps ?? "--"}
                </span>
                <span>{formatDuration(run.duration_seconds)}</span>
              </div>
              {!isRunning && (
                <button
                  type="button"
                  className="absolute right-3 top-3 rounded-md p-1 text-muted-foreground/50 opacity-0 transition-opacity hover:bg-destructive/10 hover:text-destructive group-hover:opacity-100 focus-visible:opacity-100"
                  aria-label={t("studio.history.deleteRun")}
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
            {loading ? t("studio.history.loading") : t("studio.history.loadMore")}
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
            <AlertDialogTitle>{t("studio.history.deleteTitle")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("studio.history.deleteDescription")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => void handleDelete()}
            >
              {t("common.delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
