import { Card } from "@/components/ui/card";
import { useNavigate } from "@tanstack/react-router";
import { motion } from "motion/react";
import { Suspense, lazy, useEffect, useRef, useState } from "react";

import type { ConfettiRef } from "@/components/ui/confetti";
import { STEPS } from "@/config/training";
import { useWizardStore } from "@/stores/training";
import { SplashScreen } from "./splash-screen";
import { WizardContent } from "./wizard-content";
import { WizardFooter } from "./wizard-footer";
import { WizardSidebar } from "./wizard-sidebar";

const Confetti = lazy(() =>
  import("@/components/ui/confetti").then((m) => ({ default: m.Confetti })),
);

export function WizardLayout() {
  const navigate = useNavigate();
  const [showSplash, setShowSplash] = useState(true);
  const currentStep = useWizardStore((s) => s.currentStep);
  const confettiRef = useRef<ConfettiRef>(null);
  const hasFiredRef = useRef(false);
  const isFinalStep = currentStep === STEPS.length;

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
    <div className="relative min-h-screen flex items-center justify-center p-8 bg-gradient-to-br from-primary/5 via-background to-primary/3 overflow-hidden">
      {showSplash && (
        <SplashScreen
          onStartOnboarding={() => setShowSplash(false)}
          onGoToStudio={() => navigate({ to: "/studio" })}
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
          <Card className="relative z-10 w-full !gap-0 h-[640px] flex flex-row overflow-hidden !p-0 !m-0 shadow-border ring-1 ring-border">
            <WizardSidebar />
            <div className="flex-1 flex flex-col">
              <WizardContent />
              <WizardFooter />
            </div>
          </Card>
        </motion.div>
      )}
    </div>
  );
}
