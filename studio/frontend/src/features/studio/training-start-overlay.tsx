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
import { Progress } from "@/components/ui/progress";
import {
  AnimatedSpan,
  Terminal,
  TypingAnimation,
} from "@/components/ui/terminal";
import {
  getDatasetDownloadProgress,
  getDownloadProgress,
  type DownloadProgressResponse,
} from "@/features/chat/api/chat-api";
import { useTransferStats } from "@/features/chat/hooks/use-transfer-stats";
import { formatEta, formatRate } from "@/features/chat/utils/format-transfer";
import {
  useTrainingActions,
  useTrainingConfigStore,
  useTrainingRuntimeStore,
} from "@/features/training";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState, type ReactElement } from "react";

const HF_REPO_REGEX = /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/;

function formatBytes(n: number): string {
  if (n <= 0) return "0 B";
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function formatCachePath(path: string): string {
  return path
    .replace(/^\/(?:home|Users)\/[^/]+/, "~")
    .replace(/^[A-Za-z]:[/\\]Users[/\\][^/\\]+/, "~");
}

type DownloadState = {
  downloadedBytes: number;
  totalBytes: number;
  percent: number;
  cachePath: string | null;
};

const EMPTY_DOWNLOAD_STATE: DownloadState = {
  downloadedBytes: 0,
  totalBytes: 0,
  percent: 0,
  cachePath: null,
};

function coerceCachedStateReady(state: DownloadState): DownloadState {
  if (!state.cachePath) return state;
  if (state.downloadedBytes > 0 && state.percent < 100) return state;
  const totalBytes =
    state.totalBytes > 0 ? state.totalBytes : state.downloadedBytes;
  if (totalBytes <= 0) {
    return { ...state, percent: 100 };
  }
  return {
    ...state,
    downloadedBytes: totalBytes,
    totalBytes,
    percent: 100,
  };
}

type Fetcher = (repoId: string) => Promise<DownloadProgressResponse>;

/**
 * Polls a HF repo's download progress on a 1.5s tick. Used for both
 * model weights (`/api/models/download-progress`) and dataset blobs
 * (`/api/datasets/download-progress`) by swapping the fetcher.
 *
 * Stops polling once `progress >= 1.0` -- the bar freezes at the final
 * value rather than disappearing, mirroring the existing chat flow.
 */
function useHfDownloadProgress(
  repoId: string | null,
  fetcher: Fetcher,
): DownloadState {
  const phase = useTrainingRuntimeStore((s) => s.phase);
  const isStarting = useTrainingRuntimeStore((s) => s.isStarting);
  const [state, setState] = useState<DownloadState>(EMPTY_DOWNLOAD_STATE);

  const shouldPoll =
    isStarting ||
    phase === "configuring" ||
    phase === "downloading_model" ||
    phase === "downloading_dataset" ||
    phase === "loading_model" ||
    phase === "loading_dataset" ||
    phase === "training";

  useEffect(() => {
    if (!repoId || !HF_REPO_REGEX.test(repoId) || !shouldPoll) {
      setState(EMPTY_DOWNLOAD_STATE);
      return;
    }

    let cancelled = false;
    let finished = false;
    let interval: ReturnType<typeof setInterval> | null = null;

    const poll = async () => {
      if (cancelled || finished) return;
      try {
        const prog = await fetcher(repoId);
        if (cancelled) return;
        const downloaded = prog.downloaded_bytes ?? 0;
        const total = prog.expected_bytes ?? 0;
        const ratio = prog.progress ?? 0;
        const pct =
          total > 0 ? Math.min(100, Math.round(ratio * 100)) : 0;
        setState({
          downloadedBytes: downloaded,
          totalBytes: total,
          percent: pct,
          cachePath: prog.cache_path ?? null,
        });
        if (ratio >= 1.0) {
          finished = true;
          if (interval) {
            clearInterval(interval);
            interval = null;
          }
        }
      } catch {
        // Silently swallow; bar freezes at last value (matches chat flow).
      }
    };

    void poll();
    interval = setInterval(poll, 1500);

    return () => {
      cancelled = true;
      if (interval) clearInterval(interval);
    };
  }, [repoId, shouldPoll, fetcher]);

  return state;
}

function useModelDownloadProgress(modelName: string | null): DownloadState {
  return useHfDownloadProgress(modelName, getDownloadProgress);
}

function useDatasetDownloadProgress(datasetName: string | null): DownloadState {
  return useHfDownloadProgress(datasetName, getDatasetDownloadProgress);
}

type DownloadRowProps = {
  label: string;
  state: DownloadState;
};

function DownloadRow({ label, state }: DownloadRowProps): ReactElement | null {
  // Compute a rolling-window rate + ETA from the same cumulative-byte
  // series the poll hook already produces, so we can show
  // "5.2 / 20.7 GB • 85.3 MB/s • 3m 12s left" instead of just the pair.
  const stats = useTransferStats(state.downloadedBytes, state.totalBytes);

  if (state.downloadedBytes <= 0 && !state.cachePath) return null;
  const isComplete = state.totalBytes > 0 && state.percent >= 100;
  const statusLabel = isComplete
    ? "Ready"
    : state.totalBytes > 0
      ? "Downloading"
      : state.downloadedBytes === 0
        ? "Preparing"
        : null;
  const showRate = stats.stable && !isComplete;
  const rateSuffix = showRate ? ` • ${formatRate(stats.rateBytesPerSecond)}` : "";
  const etaStr =
    showRate && state.totalBytes > 0 ? formatEta(stats.etaSeconds) : "--";
  const etaSuffix = etaStr !== "--" ? ` • ${etaStr} left` : "";
  const sizeLabel =
    state.totalBytes > 0
      ? `${formatBytes(state.downloadedBytes)} / ${formatBytes(state.totalBytes)}${rateSuffix}${etaSuffix}`
      : state.downloadedBytes > 0
        ? `${formatBytes(state.downloadedBytes)} downloaded${rateSuffix}`
        : null;
  return (
    <div className="flex flex-col gap-1.5 rounded-md border border-border/50 bg-muted/20 px-3 py-2">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-foreground/90">{label}</span>
          {statusLabel ? (
            <span
              className={`rounded-full px-1.5 py-0.5 text-[10px] font-medium ${isComplete ? "bg-emerald-100 text-emerald-700 ring-1 ring-emerald-200/80 dark:bg-emerald-500/15 dark:text-emerald-300 dark:ring-emerald-500/30" : "bg-muted text-muted-foreground"}`}
            >
              {statusLabel}
            </span>
          ) : null}
        </div>
        <span className="text-xs tabular-nums text-muted-foreground">
          {state.totalBytes > 0 ? `${state.percent}%` : ""}
        </span>
      </div>
      {sizeLabel ? (
        <div className="text-[11px] tabular-nums text-muted-foreground">
          {sizeLabel}
        </div>
      ) : null}
      {state.totalBytes > 0 ? (
        <Progress
          value={state.percent}
          indicatorClassName="bg-[linear-gradient(90deg,oklch(0.66_0.142_166.6)_0%,oklch(0.705_0.132_166.6)_55%,oklch(0.75_0.122_166.6)_100%)]"
        />
      ) : null}
      {state.cachePath ? (
        <div
          className="truncate rounded bg-muted/50 px-2 py-1 text-[10px] text-muted-foreground/70"
          title={state.cachePath}
        >
          {formatCachePath(state.cachePath)}
        </div>
      ) : null}
    </div>
  );
}

type TrainingStartOverlayProps = {
  message: string
  currentStep: number
}

export function TrainingStartOverlay({
  message,
  currentStep,
}: TrainingStartOverlayProps): ReactElement {
  const { stopTrainingRun, dismissTrainingRun } = useTrainingActions();
  const isStarting = useTrainingRuntimeStore((s) => s.isStarting);
  const phase = useTrainingRuntimeStore((s) => s.phase);
  const startModelName = useTrainingRuntimeStore((s) => s.startModelName);
  const startDatasetName = useTrainingRuntimeStore((s) => s.startDatasetName);
  const startFromResume = useTrainingRuntimeStore((s) => s.startFromResume);
  const configuredModel = useTrainingConfigStore((s) => s.selectedModel);
  const datasetSource = useTrainingConfigStore((s) => s.datasetSource);
  const dataset = useTrainingConfigStore((s) => s.dataset);
  // Only HF datasets have a download phase to track. Uploaded files are
  // already on disk by the time the overlay shows up.
  const hfDatasetName = datasetSource === "huggingface" ? dataset : null;
  const hasStartResources = startModelName !== null;
  const useConfiguredResources = !isStarting && !hasStartResources;
  const isDownloadPhase =
    phase === "downloading_model" || phase === "downloading_dataset";
  const modelName = hasStartResources
    ? startModelName
    : useConfiguredResources
      ? configuredModel
      : null;
  const datasetName = hasStartResources
    ? startDatasetName
    : useConfiguredResources
      ? hfDatasetName
      : null;
  const displayMessage =
    startFromResume && !isDownloadPhase && /^download/i.test(message)
      ? "Resuming training..."
      : message || "starting training...";
  const rawModelDownload = useModelDownloadProgress(modelName);
  const rawDatasetDownload = useDatasetDownloadProgress(datasetName);
  const modelDownload = isDownloadPhase
    ? rawModelDownload
    : coerceCachedStateReady(rawModelDownload);
  const datasetDownload = isDownloadPhase
    ? rawDatasetDownload
    : coerceCachedStateReady(rawDatasetDownload);
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false);
  const [cancelRequested, setCancelRequested] = useState(false);

  useEffect(() => {
    if (!isStarting) {
      setCancelRequested(false);
    }
  }, [isStarting]);

  return (
    <div className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center rounded-2xl bg-background/45 backdrop-blur-[1px]">
      <div className="pointer-events-auto relative flex w-[860px] max-w-[calc(100%-2rem)] flex-col items-center gap-4">
        <img
          src="/unsloth-gem.png"
          alt="Unsloth mascot"
          className="size-24 object-contain"
        />
        <div className="relative w-full">
          <AlertDialog open={cancelDialogOpen} onOpenChange={setCancelDialogOpen}>
            <Button
              variant="ghost"
              size="icon"
              className="absolute right-3 top-3 z-10 size-7 cursor-pointer rounded-full text-muted-foreground/60 hover:bg-destructive/10 hover:text-destructive"
              onClick={() => setCancelDialogOpen(true)}
              disabled={cancelRequested}
            >
              <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
            </Button>
            <AlertDialogContent overlayClassName="bg-background/40 supports-backdrop-filter:backdrop-blur-[1px]">
              <AlertDialogHeader>
                <AlertDialogTitle>Cancel Training</AlertDialogTitle>
                <AlertDialogDescription>
                  Do you want to cancel the current training run?
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Continue Training</AlertDialogCancel>
                <AlertDialogAction
                  variant="destructive"
                  onClick={() => {
                    setCancelRequested(true);
                    setCancelDialogOpen(false);
                    useTrainingRuntimeStore.getState().setStopRequested(true);
                    void stopTrainingRun(false).then((ok) => {
                      if (ok) {
                        void dismissTrainingRun();
                      } else {
                        setCancelRequested(false);
                      }
                    });
                  }}
                >
                  Cancel Training
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
          <Terminal
            className="w-full min-h-[390px] rounded-2xl px-7 py-6 text-left"
            startOnView={false}
          >
          <TypingAnimation
            duration={36}
            className="bg-gradient-to-r from-emerald-300 via-lime-300 to-teal-300 bg-clip-text font-semibold text-transparent"
          >
            {"> unsloth training starts..."}
          </TypingAnimation>
          <AnimatedSpan className="my-2">
            <pre className="whitespace-pre text-muted-foreground inline-block">{`==((====))==\n   \\\\   /|\nO^O/ \\_/ \\\n\\        /\n "-____-"`}</pre>
          </AnimatedSpan>
          <TypingAnimation duration={44}>
            {"> Preparing model and dataset..."}
          </TypingAnimation>
          <TypingAnimation duration={44}>
            {"> We are getting everything ready for your run..."}
          </TypingAnimation>
          <AnimatedSpan className="mt-2 text-muted-foreground">
            {`> ${displayMessage} | waiting for first step... (${currentStep})`}
          </AnimatedSpan>
          {datasetDownload.downloadedBytes > 0 || datasetDownload.cachePath ? (
            <AnimatedSpan className="mt-3">
              <DownloadRow
                label="Dataset"
                state={datasetDownload}
              />
            </AnimatedSpan>
          ) : null}
          {modelDownload.downloadedBytes > 0 || modelDownload.cachePath ? (
            <AnimatedSpan className="mt-3">
              <DownloadRow
                label="Model weights"
                state={modelDownload}
              />
            </AnimatedSpan>
          ) : null}
          </Terminal>
        </div>
      </div>
    </div>
  )
}
