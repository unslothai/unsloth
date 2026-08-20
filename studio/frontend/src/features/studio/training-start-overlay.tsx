


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
import { formatEta, formatRate } from "@/features/chat/utils/format-transfer";
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
import { useEffect, useState, type ReactElement } from "react";
import { useT } from "@/i18n";

const HF_REPO_REGEX = /^[A-Za-z0-9._-]+\/[A-Za-z0-9._-]+$/;

// Tracks which jobs have already played the terminal intro animation. The
// overlay unmounts on navigation away, so without this its typing/fade-in would
// replay on every return mid-run. Module-level so it survives remounts.
const animatedJobs = new Set<string>();

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

type Fetcher = (repoId: string) => Promise<DownloadProgressResponse>;

/**
 * Polls a HF repo's download progress on a 1.5s tick. Serves both model weights
 * and dataset blobs by swapping the fetcher. Stops once `progress >= 1.0`; the
 * bar freezes at the final value rather than disappearing, matching chat flow.
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
    // settling compares against the previous reading, so the poll carries it rather than reading it back out of React.
    let latest = EMPTY_DOWNLOAD_STATE;
    // the tick does not wait for the request, so a slow response can land after a newer
    // one. settling compares byte counts for equality, so a stale reading both revokes a
    // correct settle and becomes the baseline the next poll settles against -- reporting
    // the stale total as the row's size. discard anything older than what we already have.
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
        setState(next);
        // only a verified snapshot stops the tick; a settled row can still be waiting on files.
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

  return state;
}

function useModelDownloadProgress(modelName: string | null): DownloadState {
  return useHfDownloadProgress(modelName, getDownloadProgress);
}

function useDatasetDownloadProgress(datasetName: string | null): DownloadState {
  return useHfDownloadProgress(datasetName, getDatasetDownloadProgress);
}

const PROGRESS_INDICATOR_CLASS =
  "bg-[linear-gradient(90deg,var(--control-accent)_0%,color-mix(in_oklab,var(--control-accent)_72%,white)_100%)]";

type ResourceRowProps = {
  label: string;
  state: DownloadState;
  preparation: PreparationProgress | null;
};

// Whether the row would draw anything. The caller needs the same answer: the row is
// mounted unconditionally now, and its `AnimatedSpan` wrapper still lays out a line even
// when the row itself renders null, which left a blank gap in the terminal for a run whose
// dataset never produces a transfer.
export function resourceRowHasContent(
  state: DownloadState,
  preparation: PreparationProgress | null,
): boolean {
  return Boolean(preparation) || state.downloadedBytes > 0 || Boolean(state.cachePath);
}

// one row per resource for its whole setup: the transfer while bytes move, then that
// resource's preparation step once they stop.
function ResourceRow({
  label,
  state,
  preparation,
}: ResourceRowProps): ReactElement | null {
  const t = useT();
  // Rolling-window rate + ETA from the cumulative-byte series the poll hook
  // produces, so we show "5.2 / 20.7 GB • 85.3 MB/s • 3m 12s left", not just the pair.
  const stats = useTransferStats(state.downloadedBytes, state.totalBytes);

  if (!resourceRowHasContent(state, preparation)) return null;
  // the coerced state: `coerceCachedStateReady` declines to rewrite a reading with no cache
  // path, so `settled` alone put a green Ready next to a percent below 100.
  const isComplete = state.settled && state.percent >= 100;
  // gated on bytes actually moving, not on `!settled`: an orphaned `.incomplete` blob keeps
  // `downloaded !== completed` forever, which kept a processed-cache load labelled Downloading.
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
  // an unsettled transfer keeps its byte line under the preparation title: a stall and an
  // orphaned blob look alike from byte counts, so a stalled download must not lose them.
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
  const startDatasetName = useTrainingRuntimeStore((s) => s.startDatasetName);
  const startFromResume = useTrainingRuntimeStore((s) => s.startFromResume);
  const configuredModel = useTrainingConfigStore((s) => s.selectedModel);
  const datasetSource = useTrainingConfigStore((s) => s.datasetSource);
  const dataset = useTrainingConfigStore((s) => s.dataset);
  // Streaming runs never fully download the dataset (only small metadata lands
  // in the HF cache), so the cache-watching download bar would sit near 0%
  // forever and read as "stuck downloading". Show a streaming note instead.
  const datasetStreaming = useTrainingConfigStore((s) => s.datasetStreaming);
  // Only HF datasets have a download phase to track; uploaded files are already
  // on disk by the time the overlay shows up.
  const hfDatasetName = datasetSource === "huggingface" ? dataset : null;
  const hasStartResources = startModelName !== null;
  const useConfiguredResources = !isStarting && !hasStartResources;
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
    startFromResume && /^download/i.test(message)
      ? t("studio.trainingStart.resumingTraining")
      : message || t("studio.trainingStart.startingTraining");
  const rawModelDownload = useModelDownloadProgress(modelName);
  const rawDatasetDownload = useDatasetDownloadProgress(datasetName);
  const modelDownload = coerceCachedStateReady(rawModelDownload);
  const datasetDownload = coerceCachedStateReady(rawDatasetDownload);
  // the raw message, not displayMessage: a resumed run rewrites its download statuses to
  // "resuming training", which names no resource for the classifier to route on.
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

  // Play the intro animation only on the first mount per job. On later remounts
  // the terminal renders its final state instantly so the logs don't restart.
  const alreadyAnimated = jobId != null && animatedJobs.has(jobId);
  useEffect(() => {
    if (jobId != null) {
      animatedJobs.add(jobId);
    }
  }, [jobId]);

  return (
    <div className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center rounded-2xl bg-background/45 backdrop-blur-[1px]">
      <div className="pointer-events-auto relative flex w-[860px] max-w-[calc(100%-2rem)] flex-col items-center">
        <MascotImg src="unsloth-gem.png" className="size-24 object-contain" />
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
            className="w-full min-h-[390px] rounded-2xl border-0 px-7 py-6 text-left"
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
