// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MascotImg } from "@/components/mascot-img";
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
import { formatEta } from "@/features/chat/utils/format-transfer";
import { formatBytes, formatRate } from "@/features/hub";
import { useHfTokenStore } from "@/features/hub/stores/hf-token-store";
import {
  EMPTY_DOWNLOAD_STATE,
  coerceCachedStateReady,
  downloadStateFromProgress,
  type DownloadState,
} from "@/features/studio/download-state";
import {
  classifyPreparation,
  parsePreparationProgress,
  shouldShowPreparationStatus,
  type PreparationProgress,
} from "./preparation-progress";
import {
  useTrainingActions,
  useTrainingConfigStore,
  useTrainingRuntimeStore,
} from "@/features/training";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useState, type ReactElement } from "react";
import { useT } from "@/i18n";

const HF_REPO_REGEX = /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/;

// Module-level so the intro animation plays once per job across remounts.
const animatedJobs = new Set<string>();

function formatCachePath(path: string): string {
  return path
    .replace(/^\/(?:home|Users)\/[^/]+/, "~")
    .replace(/^[A-Za-z]:[/\\]Users[/\\][^/\\]+/, "~");
}

type Fetcher = (repoId: string) => Promise<DownloadProgressResponse>;

/** Stops at `progress >= 1.0`; the bar freezes rather than disappearing, as in chat. */
function useHfDownloadProgress(
  repoId: string | null,
  fetcher: Fetcher,
): DownloadState {
  const phase = useTrainingRuntimeStore((s) => s.phase);
  const isStarting = useTrainingRuntimeStore((s) => s.isStarting);
  const [state, setState] = useState<DownloadState & { repoId?: string }>(
    EMPTY_DOWNLOAD_STATE,
  );

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
    let latest = EMPTY_DOWNLOAD_STATE;
    // The tick does not wait for the request, so discard readings older than the latest; a stale one
    // would revoke a settle and become the next baseline.
    let issued = 0;
    let applied = 0;

    const poll = async () => {
      if (cancelled || finished) return;
      const generation = ++issued;
      try {
        const prog = await fetcher(repoId);
        if (cancelled || generation <= applied) return;
        applied = generation;
        const next = downloadStateFromProgress(prog, latest);
        latest = next;
        setState({ ...next, repoId });
        // Only a verified snapshot stops the tick; a settled row can still be waiting on files.
        if (next.completeOnDisk) {
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

  // Never show the old repo's Ready state while the new repo starts polling.
  return state.repoId === repoId ? state : EMPTY_DOWNLOAD_STATE;
}

function useModelDownloadProgress(
  modelName: string | null,
  hfToken: string | null,
): DownloadState {
  const fetchProgress = useCallback(
    (repoId: string) => getDownloadProgress(repoId, hfToken),
    [hfToken],
  );
  return useHfDownloadProgress(modelName, fetchProgress);
}

function useDatasetDownloadProgress(
  datasetName: string | null,
  hfToken: string | null,
): DownloadState {
  const fetchProgress = useCallback(
    (repoId: string) => getDatasetDownloadProgress(repoId, hfToken),
    [hfToken],
  );
  return useHfDownloadProgress(datasetName, fetchProgress);
}

const PROGRESS_INDICATOR_CLASS =
  "bg-[linear-gradient(90deg,var(--control-accent)_0%,color-mix(in_oklab,var(--control-accent)_72%,white)_100%)]";

type ResourceRowProps = {
  label: string;
  state: DownloadState;
  preparation: PreparationProgress | null;
};

// The caller needs this too: the AnimatedSpan wrapper lays out a line even when the row is null.
export function resourceRowHasContent(
  state: DownloadState,
  preparation: PreparationProgress | null,
): boolean {
  return Boolean(preparation) || state.downloadedBytes > 0 || Boolean(state.cachePath);
}

// One row per resource: the transfer, then its preparation step once bytes stop.
function ResourceRow({
  label,
  state,
  preparation,
}: ResourceRowProps): ReactElement | null {
  const t = useT();
  const stats = useTransferStats(state.downloadedBytes, state.totalBytes);

  if (!resourceRowHasContent(state, preparation)) return null;
  // `settled` alone can sit below 100% when coerceCachedStateReady declined to rewrite.
  const isComplete = state.settled && state.percent >= 100;
  // Gated on moving bytes, not `!settled`, since an orphaned `.incomplete` blob never settles.
  const preparing = state.moving ? null : preparation;
  const statusLabel = preparing
    ? preparing.title
    : isComplete
      ? t("studio.trainingStart.ready")
      : state.totalBytes > 0
        ? t("studio.trainingStart.downloading")
        : state.downloadedBytes === 0
          ? t("studio.trainingStart.preparing")
          : null;
  const showRate = stats.stable && !isComplete;
  const rateSuffix = showRate ? ` • ${formatRate(stats.rateBytesPerSecond)}` : "";
  const etaStr =
    showRate && state.totalBytes > 0 ? formatEta(stats.etaSeconds) : "--";
  const etaSuffix =
    etaStr !== "--" ? ` • ${t("studio.trainingStart.left", { eta: etaStr })}` : "";
  // A stall and an orphaned blob look alike, so keep the byte line under the preparation title.
  const sizeLabel = preparing
    ? (preparing.detail ??
      (state.settled || state.totalBytes <= 0
        ? null
        : `${formatBytes(state.downloadedBytes)} / ${formatBytes(state.totalBytes)}`))
    : state.totalBytes > 0
      ? `${formatBytes(state.downloadedBytes)} / ${formatBytes(state.totalBytes)}${rateSuffix}${etaSuffix}`
      : state.downloadedBytes > 0
        ? `${t("studio.trainingStart.downloaded", {
            size: formatBytes(state.downloadedBytes),
          })}${rateSuffix}`
        : null;
  const percentLabel = preparing
    ? preparing.percent !== null
      ? `${preparing.percent}%`
      : ""
    : state.totalBytes > 0
      ? `${state.percent}%`
      : "";
  return (
    <div className="flex flex-col gap-1.5 rounded-md border border-border/50 bg-muted/20 px-3 py-2">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <span className="shrink-0 text-xs text-foreground/90">{label}</span>
          {statusLabel ? (
            <span
              className={`truncate rounded-full px-1.5 py-0.5 text-ui-10 font-medium ${isComplete && !preparing ? "bg-emerald-100 text-emerald-700 ring-1 ring-emerald-200/80 dark:bg-emerald-500/15 dark:text-emerald-300 dark:ring-emerald-500/30" : "bg-muted text-muted-foreground"}`}
              title={statusLabel}
            >
              {statusLabel}
            </span>
          ) : null}
        </div>
        <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
          {percentLabel}
        </span>
      </div>
      {sizeLabel ? (
        <div className="text-ui-11 tabular-nums text-muted-foreground">
          {sizeLabel}
        </div>
      ) : null}
      {preparing ? (
        <Progress
          value={preparing.percent ?? undefined}
          indeterminate={preparing.percent === null}
          indicatorClassName={PROGRESS_INDICATOR_CLASS}
        />
      ) : state.totalBytes > 0 ? (
        <Progress
          value={state.percent}
          indicatorClassName={PROGRESS_INDICATOR_CLASS}
        />
      ) : null}
      {state.cachePath ? (
        <div
          className="truncate rounded bg-muted/50 px-2 py-1 text-ui-10 text-muted-foreground/70"
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
  const t = useT();
  const { stopTrainingRun, dismissTrainingRun } = useTrainingActions();
  const isStarting = useTrainingRuntimeStore((s) => s.isStarting);
  const phase = useTrainingRuntimeStore((s) => s.phase);
  const jobId = useTrainingRuntimeStore((s) => s.jobId);
  const startModelName = useTrainingRuntimeStore((s) => s.startModelName);
  const modelDownloadRepoId = useTrainingRuntimeStore(
    (s) => s.modelDownloadRepoId,
  );
  const startDatasetName = useTrainingRuntimeStore((s) => s.startDatasetName);
  const startHfToken = useTrainingRuntimeStore((s) => s.startHfToken);
  const startFromResume = useTrainingRuntimeStore((s) => s.startFromResume);
  const configuredModel = useTrainingConfigStore((s) => s.selectedModel);
  const configuredHfToken = useHfTokenStore((s) => s.token);
  const datasetSource = useTrainingConfigStore((s) => s.datasetSource);
  const dataset = useTrainingConfigStore((s) => s.dataset);
  // Streaming runs never fully download the dataset, so the bar would look stuck.
  const datasetStreaming = useTrainingConfigStore((s) => s.datasetStreaming);
  // Uploaded files are already on disk.
  const hfDatasetName = datasetSource === "huggingface" ? dataset : null;
  const hasStartResources = startModelName !== null;
  const useConfiguredResources = !isStarting && !hasStartResources;
  const modelName =
    modelDownloadRepoId ??
    (hasStartResources
      ? startModelName
      : useConfiguredResources
        ? configuredModel
        : null);
  const datasetName = hasStartResources
    ? startDatasetName
    : useConfiguredResources
      ? hfDatasetName
      : null;
  const hfToken = hasStartResources ? startHfToken : configuredHfToken;
  const displayMessage =
    startFromResume && /^download/i.test(message)
      ? t("studio.trainingStart.resumingTraining")
      : message || t("studio.trainingStart.startingTraining");
  const rawModelDownload = useModelDownloadProgress(modelName, hfToken);
  const rawDatasetDownload = useDatasetDownloadProgress(datasetName, hfToken);
  const modelDownload = coerceCachedStateReady(rawModelDownload);
  const datasetDownload = coerceCachedStateReady(rawDatasetDownload);
  // The raw message: a resumed run rewrites statuses to "resuming training".
  const preparationProgress = shouldShowPreparationStatus(
    phase,
    currentStep,
    isStarting,
  )
    ? parsePreparationProgress(message, t("studio.trainingStart.preparing"))
    : null;
  const preparationTarget = preparationProgress
    ? classifyPreparation(preparationProgress.title, { modelName, datasetName })
    : null;
  const datasetPreparation =
    preparationTarget === "dataset" ? preparationProgress : null;
  const modelPreparation =
    preparationTarget === "model" ? preparationProgress : null;
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false);
  const [cancelRequested, setCancelRequested] = useState(false);

  useEffect(() => {
    if (!isStarting) {
      setCancelRequested(false);
    }
  }, [isStarting]);

  const alreadyAnimated = jobId != null && animatedJobs.has(jobId);
  useEffect(() => {
    if (jobId != null) {
      animatedJobs.add(jobId);
    }
  }, [jobId]);

  // my-auto, not items-center, so a tall column never spills above the cancel button.
  return (
    <div className="pointer-events-none absolute inset-0 z-30 flex flex-col items-center rounded-2xl bg-background/45 backdrop-blur-[1px]">
      <div className="pointer-events-auto relative my-auto flex w-[calc(860px*var(--ui-space-scale,1))] max-w-[calc(100%-2rem)] flex-col items-center">
        <MascotImg src="unsloth-gem.png" className="size-24 object-contain max-sm:size-16" />
        <div className="relative w-full">
          <AlertDialog open={cancelDialogOpen} onOpenChange={setCancelDialogOpen}>
            <Button
              variant="ghost"
              size="icon"
              className="absolute right-3 top-3 z-10 size-7 cursor-pointer rounded-full text-muted-foreground/90 hover:bg-destructive/10 hover:text-destructive"
              onClick={() => setCancelDialogOpen(true)}
              disabled={cancelRequested}
            >
              <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
            </Button>
            <AlertDialogContent overlayClassName="bg-background/40 supports-backdrop-filter:backdrop-blur-[1px]">
              <AlertDialogHeader>
                <AlertDialogTitle>{t("studio.training.cancelTitle")}</AlertDialogTitle>
                <AlertDialogDescription>
                  {t("studio.training.cancelDescription")}
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>{t("studio.training.continueAction")}</AlertDialogCancel>
                <AlertDialogAction
                  variant="destructive"
                  onClick={() => {
                    setCancelRequested(true);
                    setCancelDialogOpen(false);
                    const runtime = useTrainingRuntimeStore.getState();
                    const cancellingPendingStart =
                      runtime.startRequestId !== null;
                    runtime.setStopRequested(true);
                    void stopTrainingRun(false).then((ok) => {
                      if (ok && !cancellingPendingStart) {
                        void dismissTrainingRun();
                      } else {
                        setCancelRequested(false);
                      }
                    });
                  }}
                >
                  {t("studio.training.cancelAction")}
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
          <Terminal
            className="w-full min-h-[calc(390px*var(--ui-space-scale,1))] rounded-2xl border-0 px-7 py-6 text-left max-sm:min-h-[calc(260px*var(--ui-space-scale,1))] max-sm:px-4 max-sm:py-4"
            startOnView={false}
            instant={alreadyAnimated}
          >
          <TypingAnimation
            duration={36}
            className="bg-gradient-to-r from-emerald-300 via-lime-300 to-teal-300 bg-clip-text font-semibold text-transparent"
          >
            {t("studio.trainingStart.terminalStart")}
          </TypingAnimation>
          <AnimatedSpan className="my-2">
            <pre className="whitespace-pre text-muted-foreground inline-block">{`==((====))==\n   \\\\   /|\nO^O/ \\_/ \\\n\\        /\n "-____-"`}</pre>
          </AnimatedSpan>
          <TypingAnimation duration={44}>
            {t("studio.trainingStart.preparingResources")}
          </TypingAnimation>
          <TypingAnimation duration={44}>
            {t("studio.trainingStart.gettingReady")}
          </TypingAnimation>
          <AnimatedSpan className="mt-2 text-muted-foreground">
            {t("studio.trainingStart.waitingForFirstStep", {
              message: displayMessage,
              step: currentStep,
            })}
          </AnimatedSpan>
          {datasetStreaming ? (
            <>
              <AnimatedSpan className="mt-3 text-muted-foreground">
                {t("studio.trainingStart.datasetStreaming")}
              </AnimatedSpan>
              {/* streaming has no transfer to show, but it still tokenizes and formats. the
                  row carries the preparation step on an empty state, so nothing implies a
                  download. */}
              {datasetPreparation ? (
                <AnimatedSpan className="mt-3">
                  <ResourceRow
                    label={t("studio.trainingStart.dataset")}
                    state={EMPTY_DOWNLOAD_STATE}
                    preparation={datasetPreparation}
                  />
                </AnimatedSpan>
              ) : null}
            </>
          ) : resourceRowHasContent(datasetDownload, datasetPreparation) ? (
            <AnimatedSpan className="mt-3">
              <ResourceRow
                label={t("studio.trainingStart.dataset")}
                state={datasetDownload}
                preparation={datasetPreparation}
              />
            </AnimatedSpan>
          ) : null}
          {resourceRowHasContent(modelDownload, modelPreparation) ? (
            <AnimatedSpan className="mt-3">
              <ResourceRow
                label={t("studio.trainingStart.modelWeights")}
                state={modelDownload}
                preparation={modelPreparation}
              />
            </AnimatedSpan>
          ) : null}
          </Terminal>
        </div>
      </div>
    </div>
  )
}
