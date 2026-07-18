// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  getTrainingRun,
  useTrainingConfigStore,
  useTrainingRuntimeStore,
} from "@/features/training";
import type { TrainingViewData } from "@/features/training";
import { cn } from "@/lib/utils";
import type { ReactElement } from "react";
import { useEffect, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { ChartsSection } from "./sections/charts-section";
import { ProgressSection } from "./sections/progress-section";
import {
  type RunConfigOverride,
  mapRunConfigToOverride,
} from "./sections/run-config-override";
import { TrainingStartOverlay } from "./training-start-overlay";

/** Retry budget for the run-config lookup. The run row is inserted just after
 * the progress event that reveals the run, so the first fetch can lose that
 * race; a few short retries cover the insert without polling a missing row. */
const RUN_CONFIG_FETCH_RETRIES = 5;
const RUN_CONFIG_FETCH_RETRY_MS = 1000;

/** The fetched run config only applies while it belongs to the active job;
 * a stale record from a previous run falls back to the form store. */
function activeRunOverride(
  fetched: { jobId: string; override: RunConfigOverride | undefined } | null,
  jobId: string | null,
): RunConfigOverride | undefined {
  if (fetched === null || fetched.jobId !== jobId) {
    return undefined;
  }
  return fetched.override;
}

export function LiveTrainingView(): ReactElement {
  const runtime = useTrainingRuntimeStore(
    useShallow((state) => ({
      jobId: state.jobId,
      phase: state.phase,
      message: state.message,
      error: state.error,
      currentStep: state.currentStep,
      totalSteps: state.totalSteps,
      currentEpoch: state.currentEpoch,
      currentLoss: state.currentLoss,
      currentLearningRate: state.currentLearningRate,
      currentGradNorm: state.currentGradNorm,
      currentNumTokens: state.currentNumTokens,
      progressPercent: state.progressPercent,
      elapsedSeconds: state.elapsedSeconds,
      etaSeconds: state.etaSeconds,
      evalEnabled: state.evalEnabled,
      outputDir: state.outputDir,
      isTrainingRunning: state.isTrainingRunning,
      startModelName: state.startModelName,
      startProjectName: state.startProjectName,
      lossHistory: state.lossHistory,
      lrHistory: state.lrHistory,
      gradNormHistory: state.gradNormHistory,
      evalLossHistory: state.evalLossHistory,
      firstStepReceived: state.firstStepReceived,
      isStarting: state.isStarting,
    })),
  );

  const config = useTrainingConfigStore(
    useShallow((state) => ({
      selectedModel: state.selectedModel,
      projectName: state.projectName,
      trainingMethod: state.trainingMethod,
    })),
  );

  // The Training Config popover must show the ACTIVE run's saved config, not
  // the editable form store, which the user may have changed since starting
  // the run (#6853). The backend creates the run record (with its config
  // snapshot) on the first progress event, and -- for a run that fails or
  // completes before step 1 -- from the terminal error/complete event. Fetch
  // only once one of those has happened: this avoids 404s during preparation
  // and still covers a zero-step termination, which firstStepReceived alone
  // (set only when step > 0) would miss. Until it loads (or if the fetch fails)
  // ProgressSection falls back to the form store. The fetched config is keyed by
  // job id and filtered at render time, so no synchronous reset is needed.
  // currentStep > 0 is a readiness signal in its own right: when an active run
  // is recovered through status/metrics polling (SSE unavailable or blocked),
  // applyStatus/applyMetrics restore currentStep but never set
  // firstStepReceived, so keying only off that flag would never fetch the saved
  // config for the rest of the run even though the row exists.
  const runRowReady =
    runtime.firstStepReceived ||
    runtime.currentStep > 0 ||
    runtime.phase === "completed" ||
    runtime.phase === "error" ||
    runtime.phase === "stopped";
  const [fetchedRunConfig, setFetchedRunConfig] = useState<{
    jobId: string;
    override: RunConfigOverride | undefined;
  } | null>(null);
  // Retry budget for the transient 404 below, keyed by job so a new run always
  // starts with a fresh budget.
  const [fetchAttempt, setFetchAttempt] = useState<{
    jobId: string;
    count: number;
  } | null>(null);
  useEffect(() => {
    if (!(runtime.jobId && runRowReady)) {
      return;
    }
    const jobId = runtime.jobId;
    if (fetchedRunConfig !== null && fetchedRunConfig.jobId === jobId) {
      return; // already resolved for this job
    }
    const attempts = fetchAttempt?.jobId === jobId ? fetchAttempt.count : 0;
    const controller = new AbortController();
    let retryTimer: ReturnType<typeof setTimeout> | undefined;
    getTrainingRun(jobId, controller.signal)
      .then((detail) => {
        setFetchedRunConfig({
          jobId,
          override: mapRunConfigToOverride(detail.config),
        });
      })
      .catch(() => {
        // The backend publishes the progress event before create_run commits,
        // so the row can still be missing here and this 404 is transient.
        // Nothing else in the deps changes on failure, so without an explicit
        // retry the effect would never run again for this job. Bounded, so a
        // genuinely absent row cannot poll forever -- the form-store fallback
        // simply stays in place once the budget is spent.
        if (controller.signal.aborted || attempts >= RUN_CONFIG_FETCH_RETRIES) {
          return;
        }
        retryTimer = setTimeout(() => {
          setFetchAttempt({ jobId, count: attempts + 1 });
        }, RUN_CONFIG_FETCH_RETRY_MS);
      });
    return () => {
      controller.abort();
      if (retryTimer !== undefined) {
        clearTimeout(retryTimer);
      }
    };
  }, [runtime.jobId, runRowReady, fetchedRunConfig, fetchAttempt]);
  const runConfigOverride = activeRunOverride(fetchedRunConfig, runtime.jobId);

  const activeProjectName =
    runtime.startProjectName !== null
      ? runtime.startProjectName.trim() || null
      : (config.projectName || "").trim() || null;

  const viewData: TrainingViewData = {
    phase: runtime.phase,
    currentStep: runtime.currentStep,
    totalSteps: runtime.totalSteps,
    currentLoss: runtime.currentLoss,
    currentLearningRate: runtime.currentLearningRate,
    currentGradNorm: runtime.currentGradNorm,
    currentEpoch: runtime.currentEpoch,
    currentNumTokens: runtime.currentNumTokens,
    outputDir: runtime.outputDir,
    progressPercent: runtime.progressPercent,
    elapsedSeconds: runtime.elapsedSeconds,
    etaSeconds: runtime.etaSeconds,
    evalEnabled: runtime.evalEnabled,
    message: runtime.message,
    error: runtime.error,
    isTrainingRunning: runtime.isTrainingRunning,
    modelName: runtime.startModelName ?? config.selectedModel ?? "",
    projectName: activeProjectName,
    // Prefer the saved run's method: the form may have been edited (e.g. LoRA
    // -> Full) after the run started, which would relabel the run and hide its
    // saved LoRA rows in the popover.
    trainingMethod:
      runConfigOverride?.trainingMethod ?? config.trainingMethod ?? "",
    lossHistory: runtime.lossHistory,
    lrHistory: runtime.lrHistory,
    gradNormHistory: runtime.gradNormHistory,
    evalLossHistory: runtime.evalLossHistory,
  };

  const isPreparingPhase =
    runtime.phase === "downloading_model" ||
    runtime.phase === "downloading_dataset" ||
    runtime.phase === "loading_model" ||
    runtime.phase === "loading_dataset" ||
    runtime.phase === "configuring";
  const isWaitingForFirstStep =
    runtime.phase === "training" && !runtime.firstStepReceived;
  const showOverlay =
    runtime.isStarting ||
    isPreparingPhase ||
    (isWaitingForFirstStep && runtime.currentStep <= 0);

  return (
    <div className={cn("relative", showOverlay && "min-h-[72vh]")}>
      <div
        className={cn(
          "relative z-10 flex flex-col gap-6 transition-[filter]",
          showOverlay && "blur",
        )}
      >
        <div data-tour="studio-training-progress">
          <ProgressSection
            key={runtime.jobId ?? "no-job"}
            data={viewData}
            configOverride={runConfigOverride}
          />
        </div>
        <ChartsSection
          currentStep={viewData.currentStep}
          totalSteps={viewData.totalSteps}
          isTraining={viewData.isTrainingRunning}
          evalEnabled={viewData.evalEnabled}
          lossHistory={viewData.lossHistory}
          lrHistory={viewData.lrHistory}
          gradNormHistory={viewData.gradNormHistory}
          evalLossHistory={viewData.evalLossHistory}
        />
      </div>
      {showOverlay ? (
        <TrainingStartOverlay
          message={runtime.message}
          currentStep={runtime.currentStep}
        />
      ) : null}
    </div>
  );
}
