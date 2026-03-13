// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { useTrainingRuntimeStore } from "@/features/training";
import type { ReactElement } from "react";
import { useShallow } from "zustand/react/shallow";
import { ChartsSection } from "./sections/charts-section";
import { ProgressSection } from "./sections/progress-section";
import { TrainingStartOverlay } from "./training-start-overlay";

export function TrainingView(): ReactElement {
  const runtime = useTrainingRuntimeStore(
    useShallow((state) => ({
      phase: state.phase,
      message: state.message,
      currentStep: state.currentStep,
      firstStepReceived: state.firstStepReceived,
      isStarting: state.isStarting,
    })),
  );

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
          <ProgressSection />
        </div>
        <ChartsSection />
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
