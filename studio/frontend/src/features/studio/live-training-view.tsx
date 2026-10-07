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

/** A lookup issued as the run starts can briefly miss the row. */
const RUN_CONFIG_FETCH_RETRIES = 5;
const RUN_CONFIG_FETCH_RETRY_MS = 1000;

/** A stale record from a previous run falls back to the form store. */
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
      warnings: state.warnings,
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
      sessionStartStep: state.sessionStartStep,
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
      modelType: state.modelType,
    })),
  );

  // Show the active run's saved config, not the editable form store.
  const [fetchedRunConfig, setFetchedRunConfig] = useState<{
    jobId: string;
    override: RunConfigOverride | undefined;
  } | null>(null);
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
      return;
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
        // Nothing in the deps changes on failure, so retry explicitly; bounded for a truly absent row.
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
    sessionStartStep: runtime.sessionStartStep,
    evalEnabled: runtime.evalEnabled,
    message: runtime.message,
    error: runtime.error,
    warnings: runtime.warnings,
    isTrainingRunning: runtime.isTrainingRunning,
    modelName: runtime.startModelName ?? config.selectedModel ?? "",
    projectName: activeProjectName,
    // The form may have been edited since start (e.g. LoRA -> Full).
    trainingMethod:
      runConfigOverride?.trainingMethod ?? config.trainingMethod ?? "",
    isDecision:
      runConfigOverride?.isDecision ?? config.modelType === "decision",
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
    <div className={cn("relative", showOverlay && "min-h-[72dvh]")}>
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
