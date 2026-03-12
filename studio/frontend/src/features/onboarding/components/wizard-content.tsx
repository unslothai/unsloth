// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { STEPS } from "@/config/training";
import { useTrainingConfigStore } from "@/features/training";
import type { StepNumber } from "@/types/training";
import { DatasetStep } from "./steps/dataset-step";
import { HyperparametersStep } from "./steps/hyperparameters-step";
import { ModelSelectionStep } from "./steps/model-selection-step";
import { ModelTypeStep } from "./steps/model-type-step";
import { SummaryStep } from "./steps/summary-step";

const STEP_COMPONENTS = {
  1: ModelTypeStep,
  2: ModelSelectionStep,
  3: DatasetStep,
  4: HyperparametersStep,
  5: SummaryStep,
} as const;

const STEP_MASCOTS: Record<StepNumber, string> = {
  1: "/Sloth emojis/large sloth wave.png",
  2: "/Sloth emojis/sloth magnify final.png",
  3: "/Sloth emojis/sloth huglove large.png",
  4: "/Sloth emojis/large sloth glasses.png",
  5: "/Sloth emojis/large sloth yay.png",
};

export function WizardContent() {
  const currentStep = useTrainingConfigStore((s) => s.currentStep);
  const stepConfig = STEPS[currentStep - 1];
  const StepComponent = STEP_COMPONENTS[currentStep];
  const mascotSrc = STEP_MASCOTS[currentStep];

  return (
    <main className="flex-1 flex flex-col overflow-y-auto">
      <header className="flex flex-wrap items-start gap-3 p-4 pb-3 sm:p-6 sm:pb-4">
        <img src={mascotSrc} alt="Unsloth mascot" className="size-12 sm:size-14" />
        <div className="flex flex-col min-w-0">
          <h1 className="text-lg font-semibold sm:text-xl">{stepConfig.title}</h1>
          <p className="text-sm text-muted-foreground">
            {stepConfig.description}
          </p>
        </div>
        <p className="ml-auto hidden shrink-0 text-xs text-muted-foreground uppercase tracking-wider md:block">
          Step {currentStep} of {STEPS.length}
        </p>
      </header>
      <div className="flex-1 p-4 pt-1.5 sm:p-6 sm:pt-2">
        <StepComponent />
      </div>
    </main>
  );
}
