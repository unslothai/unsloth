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

/** Retry budget for the run-config lookup. The row is inserted at
 * start_training(), but a lookup issued in the same instant can still miss it;
 * a few short retries cover that without polling a genuinely absent row. */
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

  // Show the ACTIVE run's saved config, not the editable form store the user may
  // have changed since starting (#6853). start_training() commits the run row
  // before the pump, so the job id alone gates the fetch; the bounded retry below
  // covers the narrow uncommitted window, and until it loads ProgressSection falls
  // back to the form store. The result is keyed by job id and filtered at render.
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
    if (!runtime.jobId) {
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
        // A lookup racing the row commit can miss transiently; nothing else in
        // the deps changes on failure, so retry explicitly. Bounded so a genuinely
        // absent row falls back to the form store instead of polling forever.
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
  }, [runtime.jobId, fetchedRunConfig, fetchAttempt]);
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
