// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import {
  useTrainingConfigStore,
  useTrainingRuntimeStore,
} from "@/features/training";
import type { TrainingViewData } from "@/features/training";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { type ReactElement, useCallback, useEffect, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { ChartsSection } from "./sections/charts-section";
import { NeuronHeatmapSection, ReplayControls } from "./sections/neuron-heatmap-section";
import { NeuronHealthTrend } from "./sections/neuron-health-trend";
import { DiagnosticsPanel } from "./sections/diagnostics-panel";
import { ProgressSection } from "./sections/progress-section";
import { TrainingStartOverlay } from "./training-start-overlay";
import { useActivationData } from "@/features/training/hooks/use-activation-data";

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
      <div className="h-[280px]">
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
      trainingMethod: state.trainingMethod,
      enableActivationCapture: state.enableActivationCapture,
      dataset: state.dataset,
    })),
  );

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
    modelName: config.selectedModel ?? "",
    datasetName: config.dataset ?? null,
    trainingMethod: config.trainingMethod ?? "",
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
          "relative z-10 flex flex-col gap-4 transition-[filter]",
          showOverlay && "blur",
        )}
      >
        <div data-tour="studio-training-progress">
          <ProgressSection key={runtime.jobId ?? "no-job"} data={viewData} />
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
              <div className="flex flex-col items-center justify-center min-h-[300px] gap-3 text-center text-muted-foreground">
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
