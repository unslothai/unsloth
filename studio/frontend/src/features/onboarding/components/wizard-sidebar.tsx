// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { STEPS } from "@/config/training";
import { markOnboardingDone } from "@/features/auth";
import { useTrainingConfigStore } from "@/features/training";
import { ArrowRight02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { WizardStepItem } from "./wizard-step-item";

export function WizardSidebar({ returnTo }: { returnTo: string }) {
  const currentStep = useTrainingConfigStore((s) => s.currentStep);
  const progress = ((currentStep - 1) / (STEPS.length - 1)) * 100;

  return (
    <aside className="w-full shrink-0 bg-muted/70 p-4 md:w-64 md:p-6">
      <div className="flex items-center gap-3 py-1 md:py-2">
        <img
          src="https://unsloth.ai/cgi/image/unsloth_sticker_no_shadow_ldN4V4iydw00qSIIWDCUv.png?width=96&quality=80&format=auto"
          alt="Unsloth"
          className="size-12"
        />
        <div className="flex flex-col">
          <span className="font-semibold text-lg leading-tight">Unsloth</span>
          <span className="text-xs text-muted-foreground">Studio</span>
        </div>
      </div>
      <div className="mt-3 md:mt-0">
        <Progress value={progress} className="h-1.5" />
      </div>
      <p className="mt-2 text-xs text-muted-foreground md:hidden">
        Step {currentStep} of {STEPS.length}
      </p>
      <Button
        size="sm"
        className="mt-2 w-full md:hidden"
        onClick={() => {
          markOnboardingDone();
          window.location.assign(returnTo);
        }}
      >
        Skip onboarding
        <HugeiconsIcon icon={ArrowRight02Icon} data-icon="inline-end" />
      </Button>
      <nav className="mt-3 hidden flex-col gap-1 md:flex">
        {STEPS.map((step) => (
          <WizardStepItem key={step.number} step={step} />
        ))}
      </nav>
      <Button
        size="sm"
        className="mt-3 hidden w-full md:flex"
        onClick={() => {
          markOnboardingDone();
          window.location.assign(returnTo);
        }}
      >
        Skip onboarding
        <HugeiconsIcon icon={ArrowRight02Icon} data-icon="inline-end" />
      </Button>
    </aside>
  );
}
