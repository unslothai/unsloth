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
import { Checkbox } from "@/components/ui/checkbox";
import { Spinner } from "@/components/ui/spinner";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { bumpInventoryVersion } from "@/features/hub";
import type { TrainingRunSummary } from "@/features/training";
import {
  HistoryRequestError,
  deleteTrainingRun,
  emitTrainingRunDeleted,
  getTrainingRunDisplayTitle,
  getTrainingRunModelSubtitle,
  listTrainingRuns,
  onTrainingRunDeleted,
  onTrainingRunUpdated,
  onTrainingRunsChanged,
  shouldShowTrainingArtifactsDeleted,
  useTrainingActions,
} from "@/features/training";
import { translate, useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { formatDuration } from "./sections/progress-section-lib";

type StudioT = ReturnType<typeof useT>;

const PAGE_SIZE = 12;
const RUNNING_POLL_INTERVAL_MS = 5000;

const statusBadge: Record<string, { className: string }> = {
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
    className: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-400",
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

function sharesOutputDirInVisibleRuns(
  run: TrainingRunSummary,
  runs: TrainingRunSummary[],
): boolean {
  if (!run.output_dir) return false;
  return runs.some(
    (other) => other.id !== run.id && other.output_dir === run.output_dir,
  );
}

function catmullRomPath(points: { x: number; y: number }[]): string {
  if (points.length < 2) return "";
  const firstPoint = points[0];
  if (!firstPoint) return "";
  const d = [`M${firstPoint.x.toFixed(1)},${firstPoint.y.toFixed(1)}`];
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[Math.max(i - 1, 0)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(i + 2, points.length - 1)];
    if (!p0 || !p1 || !p2 || !p3) break;
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
  const firstValue = values[0];
  if (firstValue === undefined) return null;
  let min = firstValue;
  let max = firstValue;
  for (let i = 1; i < values.length; i++) {
    const value = values[i];
    if (value === undefined) continue;
    if (value < min) min = value;
    if (value > max) max = value;
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
  const last = pts.at(-1);
  const first = pts[0];
  if (!last || !first) return null;
  const fillPath = `${linePath} L${last.x.toFixed(1)},${h} L${first.x.toFixed(1)},${h} Z`;

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="h-8 w-full"
      preserveAspectRatio="none"
      role="img"
      aria-label={ariaLabel}
    >
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

function hasTextSelectionWithin(element: HTMLElement): boolean {
  const selection = window.getSelection();
  return (
    !!selection &&
    !selection.isCollapsed &&
    element.contains(selection.anchorNode) &&
    element.contains(selection.focusNode)
  );
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
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deleteArtifacts, setDeleteArtifacts] = useState(false);
  const [resumeTarget, setResumeTarget] = useState<string | null>(null);
  const [manualFetchInFlight, setManualFetchInFlight] = useState(false);
  const { resumeTrainingRunFromHistory, startPending } = useTrainingActions();
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

  const fetchRuns = useCallback(
    async (offset = 0, append = false, limit = PAGE_SIZE) => {
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
    },
    [],
  );

  useEffect(() => {
    const timer = window.setTimeout(() => {
      void fetchRuns(0);
    }, 0);
    return () => {
      window.clearTimeout(timer);
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
    try {
      const result = await deleteTrainingRun(deleteTarget, { deleteArtifacts });
      if (deleteArtifacts) {
        bumpInventoryVersion();
        if (result.artifacts_kept_reason === "shared_output_dir") {
          toast.info(translate("studio.history.artifactsKeptShared"));
        } else if (!result.artifacts_deleted) {
          toast.error(translate("studio.history.deleteArtifactsFailed"));
        }
      }
      emitTrainingRunDeleted(deleteTarget);
      // Re-fetch preserving visible count so "Load more" offsets stay consistent.
      const currentCount = runs.length - 1;
      const limit = Math.max(PAGE_SIZE, currentCount);
      fetchRuns(0, false, limit).catch(() => {
        // Refresh failed; card is already removed, no stale display.
      });
    } catch (err) {
      toast.error(
        err instanceof HistoryRequestError &&
          err.status === 409 &&
          deleteArtifacts
          ? translate("studio.history.deleteArtifactsActiveError")
          : translate("studio.history.deleteError"),
      );
    }
    setDeleteTarget(null);
    setDeleteArtifacts(false);
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

  const deleteTargetRun = runs.find((run) => run.id === deleteTarget);
  const deleteTargetShared =
    !!deleteTargetRun &&
    (deleteTargetRun.resumed_later ||
      sharesOutputDirInVisibleRuns(deleteTargetRun, runs));

  return (
    <div className="contents" aria-label={t("studio.history.title")}>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {runs.map((run) => {
          const wasContinued =
            run.resumed_later || wasContinuedInVisibleRuns(run, runs);
          const badge = wasContinued
            ? statusBadge.resumed_later
            : (statusBadge[run.status] ?? statusBadge.error);
          const isRunning = run.status === "running";
          const artifactsMissing = shouldShowTrainingArtifactsDeleted(run);
          const canResume =
            run.can_resume && !wasContinued && !artifactsMissing;
          const isResuming = resumeTarget === run.id;

          const title = getTrainingRunDisplayTitle(run);
          const modelSubtitle = getTrainingRunModelSubtitle(run);
          const cardId = `training-run-${encodeURIComponent(run.id)}`;

          const projectSubtitle =
            run.project_name && title !== run.project_name
              ? run.project_name
              : null;
          // Backend /p ref + its capability token. Both are required: the link
          // is useless (404s) without the signature, so don't offer to copy it.
          const canCopyPreview = !!run.preview_ref && !!run.preview_sig;
          return (
            <div
              key={run.id}
              className={cn(
                "elevated-card group relative h-[11.5rem] cursor-pointer bg-card transition-colors hover:bg-accent/30",
                isRunning && "!border-blue-400/50 dark:!border-blue-500/30",
              )}
            >
              <button
                type="button"
                aria-labelledby={`${cardId}-title`}
                aria-describedby={`${cardId}-status ${cardId}-details ${cardId}-metrics`}
                className={cn(
                  "relative z-0 flex h-full w-full cursor-pointer select-text flex-col gap-3 rounded-[inherit] border-0 bg-transparent p-4 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                  (canResume || canCopyPreview) && "gap-2",
                )}
                onClick={(event) => {
                  if (
                    event.detail > 0 &&
                    hasTextSelectionWithin(event.currentTarget)
                  ) {
                    return;
                  }
                  onSelectRun(run.id);
                }}
              >
                <span
                  id={`${cardId}-status`}
                  className="flex items-center justify-between pr-6"
                >
                  <span
                    className={cn(
                      "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-ui-10 font-semibold",
                      badge.className,
                    )}
                  >
                    {isRunning && <Spinner className="size-2.5" />}
                    {formatStatusLabel(
                      wasContinued ? "resumed_later" : run.status,
                      t,
                    )}
                  </span>
                  {artifactsMissing && (
                    <span className="inline-flex items-center rounded-full bg-foreground/[0.05] px-2 py-0.5 text-ui-10 font-medium text-muted-foreground dark:bg-white/[0.06]">
                      {t("studio.history.filesDeleted")}
                    </span>
                  )}
                  <span className="text-ui-10 text-muted-foreground">
                    {formatRelativeTime(run.started_at, t)}
                  </span>
                </span>
                <span className="block min-w-0 cursor-text">
                  <span
                    id={`${cardId}-title`}
                    className="block truncate text-sm font-medium"
                    title={title}
                  >
                    {title}
                  </span>
                  <span id={`${cardId}-details`} className="block">
                    {modelSubtitle && (
                      <span
                        className="block truncate text-xs text-muted-foreground"
                        title={modelSubtitle}
                      >
                        {modelSubtitle}
                      </span>
                    )}
                    <span
                      className="block truncate text-xs text-muted-foreground"
                      title={run.dataset_name}
                    >
                      {run.dataset_name}
                    </span>
                    {projectSubtitle && (
                      <span
                        className="block truncate text-xs text-muted-foreground/80"
                        title={projectSubtitle}
                      >
                        {projectSubtitle}
                      </span>
                    )}
                  </span>
                </span>
                {run.loss_sparkline && run.loss_sparkline.length >= 2 && (
                  <span
                    className={cn(
                      "block",
                      (canResume || canCopyPreview) && "h-7 overflow-hidden",
                    )}
                  >
                    <Sparkline
                      values={run.loss_sparkline}
                      id={run.id}
                      ariaLabel={t("studio.history.lossTrendSparkline")}
                    />
                  </span>
                )}
                <span
                  id={`${cardId}-metrics`}
                  className="flex flex-wrap gap-x-4 gap-y-1 text-ui-11 text-muted-foreground"
                >
                  <span>
                    {t("studio.history.loss")}:{" "}
                    {run.final_loss != null ? run.final_loss.toFixed(4) : "--"}
                  </span>
                  <span>
                    {t("studio.history.steps")}: {run.final_step ?? 0}/
                    {run.total_steps ?? "--"}
                  </span>
                  <span>{formatDuration(run.duration_seconds)}</span>
                </span>
              </button>
              {canResume && (
                <Button
                  type="button"
                  size="xs"
                  variant="outline"
                  className="absolute bottom-3 left-4 z-10 h-6 rounded-full px-2.5 text-ui-11 leading-none shadow-sm"
                  disabled={startPending || isResuming}
                  onClick={() => void handleResume(run.id)}
                >
                  {isResuming
                    ? t("studio.history.resuming")
                    : t("studio.history.resumeTraining")}
                </Button>
              )}
              {canCopyPreview && (
                <Button
                  type="button"
                  size="xs"
                  variant="outline"
                  className="absolute bottom-3 right-4 z-10 h-6 rounded-full px-2.5 text-ui-11 leading-none shadow-sm"
                  onClick={async () => {
                    const ref = (run.preview_ref ?? "")
                      .split("/")
                      .map(encodeURIComponent)
                      .join("/");
                    const base = (
                      cloudflareUrl ??
                      serverUrl ??
                      window.location.origin
                    ).replace(/\/+$/, "");
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
              {!isRunning && (
                <button
                  type="button"
                  className="absolute right-3 top-3 z-10 rounded-md p-1 text-muted-foreground/50 opacity-0 transition-opacity hover:bg-destructive/10 hover:text-destructive group-hover:opacity-100 focus-visible:opacity-100"
                  aria-label={t("studio.history.deleteRun")}
                  onClick={() => setDeleteTarget(run.id)}
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
            {loading
              ? t("studio.history.loading")
              : t("studio.history.loadMore")}
          </Button>
        </div>
      )}
      {loading && runs.length === 0 && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {["first", "second", "third"].map((skeletonId) => (
            <div
              key={`skeleton-${skeletonId}`}
              className="h-40 animate-pulse rounded-xl border bg-muted/30"
            />
          ))}
        </div>
      )}
      <AlertDialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) {
            setDeleteTarget(null);
            setDeleteArtifacts(false);
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t("studio.history.deleteTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t("studio.history.deleteDescription")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          {deleteTargetRun?.artifacts_available === true && (
            <label
              htmlFor="delete-run-artifacts"
              className="flex cursor-pointer items-start gap-2 text-sm"
            >
              <Checkbox
                id="delete-run-artifacts"
                checked={deleteArtifacts}
                onCheckedChange={(value) => setDeleteArtifacts(!!value)}
                className="mt-0.5"
              />
              <span className="flex flex-col gap-0.5">
                <span className="text-foreground">
                  {t("studio.history.deleteArtifactsLabel")}
                </span>
                <span className="text-xs text-muted-foreground">
                  {t("studio.history.deleteArtifactsDescription")}
                </span>
                {deleteTargetShared && (
                  <span className="text-xs text-amber-600 dark:text-amber-400">
                    {t("studio.history.deleteArtifactsSharedNote")}
                  </span>
                )}
              </span>
            </label>
          )}
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
