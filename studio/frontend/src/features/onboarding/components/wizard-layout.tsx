// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Card } from "@/components/ui/card";
import { Route as OnboardingRoute } from "@/app/routes/onboarding";
import { motion } from "motion/react";
import { Suspense, lazy, useEffect, useRef, useState } from "react";

import type { ConfettiRef } from "@/components/ui/confetti";
import { STEPS } from "@/config/training";
import { isOnboardingDone, markOnboardingDone } from "@/features/auth";
import { useTrainingConfigStore } from "@/features/training";
import { SplashScreen } from "./splash-screen";
import { WizardContent } from "./wizard-content";
import { WizardFooter } from "./wizard-footer";
import { WizardSidebar } from "./wizard-sidebar";

const Confetti = lazy(() =>
  import("@/components/ui/confetti").then((m) => ({ default: m.Confetti })),
);

function sanitizeRedirectTarget(value: string | undefined): string {
  if (!value) return "/chat";
  if (!value.startsWith("/")) return "/chat";
  if (value.startsWith("//")) return "/chat";
  if (value.includes("\\")) return "/chat";
  return value;
}

export function WizardLayout() {
  const search = OnboardingRoute.useSearch();
  const [showSplash, setShowSplash] = useState(true);
  const currentStep = useTrainingConfigStore((s) => s.currentStep);
  const confettiRef = useRef<ConfettiRef>(null);
  const hasFiredRef = useRef(false);
  const isFinalStep = currentStep === STEPS.length;
  const returnTo = sanitizeRedirectTarget(search.redirectTo);
  const exitToReturnTo = () => window.location.assign(returnTo);

  // Only redirect on initial mount — not on re-renders after markOnboardingDone()
  // which would override explicit /chat navigation from skip buttons.
  const checkedRef = useRef(false);
  useEffect(() => {
    if (!checkedRef.current) {
      checkedRef.current = true;
      if (isOnboardingDone()) {
        exitToReturnTo();
      }
    }
  }, [returnTo]);

  useEffect(() => {
    if (isFinalStep && !hasFiredRef.current) {
      hasFiredRef.current = true;
      confettiRef.current?.fire({
        particleCount: 80,
        angle: 60,
        spread: 55,
        origin: { x: 0, y: 0.6 },
        colors: ["#34b482", "#26ccff", "#a25afd", "#88ff5a"],
      });
      confettiRef.current?.fire({
        particleCount: 80,
        angle: 120,
        spread: 55,
        origin: { x: 1, y: 0.6 },
        colors: ["#34b482", "#26ccff", "#a25afd", "#88ff5a"],
      });
    }
    if (!isFinalStep) {
      hasFiredRef.current = false;
    }
  }, [isFinalStep]);

  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden bg-gradient-to-br from-primary/5 via-background to-primary/3 p-4 sm:p-6 md:p-8">
      {showSplash && (
        <SplashScreen
          onStartOnboarding={() => setShowSplash(false)}
          onSkipOnboarding={() => {
            markOnboardingDone();
            exitToReturnTo();
          }}
        />
      )}
      <Suspense fallback={null}>
        <Confetti
          ref={confettiRef}
          manualstart={true}
          className="pointer-events-none fixed inset-0 z-50 size-full"
        />
      </Suspense>
      {!showSplash && (
        <motion.div
          className="w-full max-w-5xl"
          initial={{ opacity: 0, scale: 0.98, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{
            duration: 0.4,
            ease: [0.165, 0.84, 0.44, 1],
          }}
        >
          <Card className="relative z-10 w-full !gap-0 !m-0 !p-0 flex min-h-[560px] flex-col overflow-hidden shadow-border ring-1 ring-border md:min-h-[620px] md:flex-row lg:h-[660px]">
            <WizardSidebar returnTo={returnTo} />
            <div className="flex-1 flex flex-col">
              <WizardContent />
              <WizardFooter returnTo={returnTo} onBackToSplash={() => setShowSplash(true)} />
            </div>
          </Card>
        </motion.div>
      )}
    </div>
  );
}
