// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { STEPS } from "@/config/training";
import { markOnboardingDone } from "@/features/auth";
import { useTrainingConfigStore } from "@/features/training";
import { ArrowLeft02Icon, ArrowRight02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useShallow } from "zustand/react/shallow";

export function WizardFooter({
  returnTo,
  onBackToSplash,
}: {
  returnTo: string;
  onBackToSplash: () => void;
}) {
  const { currentStep, prevStep, nextStep, canProceed } = useTrainingConfigStore(
    useShallow((s) => ({
      currentStep: s.currentStep,
      prevStep: s.prevStep,
      nextStep: s.nextStep,
      canProceed: s.canProceed(),
    })),
  );
  const isFirst = currentStep === 1;
  const isLast = currentStep === STEPS.length;

  return (
    <footer>
      <div className="flex items-center justify-between p-6">
        <Button
          variant="outline"
          className="px-4 !pl-4"
          onClick={isFirst ? onBackToSplash : prevStep}
        >
          <HugeiconsIcon icon={ArrowLeft02Icon} data-icon="inline-start" />
          Back
        </Button>
        <div className="flex items-center gap-2">
          {!isLast && (
            <Button
              variant="outline"
              className="px-4"
              onClick={() => {
                markOnboardingDone();
                window.location.assign(returnTo);
              }}
            >
              Skip
            </Button>
          )}
          {isLast ? (
            <Button
              onClick={() => {
                markOnboardingDone();
                window.location.assign(returnTo);
              }}
              disabled={!canProceed}
              className="px-4 !pr-4"
            >
              Finish onboarding
              <HugeiconsIcon icon={ArrowRight02Icon} data-icon="inline-end" />
            </Button>
          ) : (
            <Button
              onClick={() => {
                if (currentStep === 1 && sessionStorage.getItem("unsloth_chat_only") === "1") {
                  sessionStorage.removeItem("unsloth_chat_only");
                  markOnboardingDone();
                  window.location.assign("/chat");
                } else {
                  nextStep();
                }
              }}
              className="px-4 !pl-4"
              disabled={!canProceed}
            >
              Continue
              <HugeiconsIcon icon={ArrowRight02Icon} data-icon="inline-end" />
            </Button>
          )}
        </div>
      </div>
    </footer>
  );
}
