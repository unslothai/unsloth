// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { useTrainingConfigStore } from "@/features/training";
import type { StepConfig, StepNumber } from "@/types/training";
import { useShallow } from "zustand/react/shallow";

interface WizardStepItemProps {
  step: StepConfig;
}

export function WizardStepItem({ step }: WizardStepItemProps) {
  const { currentStep, setStep } = useTrainingConfigStore(
    useShallow((s) => ({ currentStep: s.currentStep, setStep: s.setStep })),
  );
  const isActive = currentStep === step.number;
  const isCompleted = currentStep > step.number;
  const canClick = isCompleted;

  return (
    <button
      type="button"
      onClick={() => canClick && setStep(step.number as StepNumber)}
      disabled={!canClick}
      className={cn(
        "flex items-start gap-3 text-left w-full py-2 transition-colors",
        canClick && "cursor-pointer hover:opacity-80",
        !(canClick || isActive) && "opacity-50",
      )}
    >
      <div
        className={cn(
          "size-5 rounded-full flex items-center justify-center text-xs font-medium shrink-0 mt-0.5 transition-colors",
          isActive && "bg-primary text-primary-foreground",
          isCompleted && "bg-primary/20 text-primary",
          !(isActive || isCompleted) && "bg-muted text-muted-foreground",
        )}
      >
        {isCompleted ? "✓" : step.number}
      </div>
      <div className="flex flex-col gap-1">
        <span
          className={cn(
            "text-sm font-medium",
            isActive && "text-foreground",
            !isActive && "text-muted-foreground",
          )}
        >
          {step.title}
        </span>
        <span className="text-xs text-muted-foreground">{step.subtitle}</span>
      </div>
    </button>
  );
}
