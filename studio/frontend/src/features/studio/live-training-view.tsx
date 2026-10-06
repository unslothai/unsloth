// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  getTrainingRun,
  useActivationData,
  useTrainingConfigStore,
  useTrainingRuntimeStore,
} from "@/features/training";
import type { TrainingViewData } from "@/features/training";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { type ReactElement, useCallback, useEffect, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { ChartsSection } from "./sections/charts-section";
import { NeuronHeatmapSection, ReplayControls } from "./sections/neuron-heatmap-section";
import { NeuronHealthTrend } from "./sections/neuron-health-trend";
import { DiagnosticsPanel } from "./sections/diagnostics-panel";
import { ProgressSection } from "./sections/progress-section";
import {
  type RunConfigOverride,
  mapRunConfigToOverride,
} from "./sections/run-config-override";
import { TrainingStartOverlay } from "./training-start-overlay";

function InterpretabilitySection({
  isTraining,
  jobId,
}: {
  isTraining: boolean;
  jobId: string | null;
}): ReactElement {
  const { metadata, records, loading } = useActivationData({ isTraining, jobId });
  const [stepIndex, setStepIndex] = useState<number>(0);

  // Keep step index at the latest record after training finishes
  useEffect(() => {
    if (!isTraining) setStepIndex(Math.max(0, records.length - 1));
  }, [records.length, isTraining]);

  const handleStepChange = useCallback(
    (idx: number) => {
      if (idx === -1) {
        setStepIndex((prev) => Math.min(prev + 1, records.length - 1));
      } else {
        setStepIndex(Math.max(0, Math.min(idx, records.length - 1)));
      }
    },
    [records.length],
  );

  // During training always show the latest; after training the slider drives it
  const displayIndex = isTraining ? Math.max(0, records.length - 1) : stepIndex;
  const record = records[displayIndex] ?? null;

  return (
    <div className="flex flex-col gap-4">
      {/* Heatmap — full width, horizontal */}
      <NeuronHeatmapSection
        isTraining={isTraining}
        records={records}
        metadata={metadata}
        loading={loading}
        record={record}
        stepIndex={displayIndex}
        onStepChange={handleStepChange}
      />

      {/* Trend chart — full width, compact */}
      <div className="h-[calc(280px*var(--ui-space-scale,1))]">
        <NeuronHealthTrend
          records={records}
          stepIndex={displayIndex}
          onStepChange={handleStepChange}
        />
      </div>

      {/* Replay controls */}
      {!isTraining && records.length > 1 && (
        <ReplayControls
          stepIndex={stepIndex}
          totalSteps={records.length}
          onStepChange={handleStepChange}
          currentStep={record?.step ?? 0}
        />
      )}

      {/* Diagnostics */}
      {records.length > 0 && (
        <DiagnosticsPanel
          records={records}
          stepIndex={displayIndex}
          metadata={metadata}
          onStepChange={handleStepChange}
        />
      )}
    </div>
  );
}

/** Retry budget for the run-config lookup. The row is inserted at start_training(), but a
* lookup issued in the same instant can still miss it; a few short retries cover that. */
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
      enableActivationCapture: state.enableActivationCapture,
      dataset: state.dataset,
    })),
  );

  // Show the ACTIVE run's saved config, not the editable form store the user may have changed
  // since starting (#6853). start_training() commits the run row before the pump, so the job id
  // alone gates the fetch; the bounded retry below covers the uncommitted window, and until it
  // loads ProgressSection falls back to the form store.
  const [fetchedRunConfig, setFetchedRunConfig] = useState<{
    jobId: string;
    override: RunConfigOverride | undefined;
  } | null>(null);
  // Retry budget for the transient 404 below, keyed by job so a new run starts fresh.
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
        // A lookup racing the row commit can miss transiently, and nothing else in the deps changes on
        // failure, so retry explicitly. Bounded so a genuinely absent row falls back to the form store.
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
    datasetName: config.dataset ?? null,
    // Prefer the saved run's method: the form may have been edited (e.g. LoRA -> Full) after the
    // run started, which would relabel the run and hide its saved LoRA rows in the popover.
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
    <div className={cn("relative", showOverlay && "min-h-[72dvh]")}>
      <div
        className={cn(
          "relative z-10 flex flex-col gap-4 transition-[filter]",
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
        <Tabs defaultValue="training">
          <TabsList className="mb-2">
            <TabsTrigger value="training">Training</TabsTrigger>
            <TabsTrigger value="interpretability">Interpretability</TabsTrigger>
          </TabsList>

          <TabsContent value="training">
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
          </TabsContent>

          <TabsContent value="interpretability">
            {config.enableActivationCapture ? (
              <InterpretabilitySection
                isTraining={viewData.isTrainingRunning}
                jobId={runtime.jobId}
              />
            ) : (
              <div className="flex flex-col items-center justify-center min-h-[calc(300px*var(--ui-space-scale,1))] gap-3 text-center text-muted-foreground">
                <p className="text-sm">Neuron activation capture is disabled.</p>
                <p className="text-xs max-w-sm">
                  Enable <span className="font-medium text-foreground">Neuron activation capture</span> in
                  training parameters before starting training to see interpretability data here.
                </p>
              </div>
            )}
          </TabsContent>
        </Tabs>
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
