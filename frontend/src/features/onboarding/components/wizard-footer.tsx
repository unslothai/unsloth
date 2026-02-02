import { Button } from "@/components/ui/button";
import { STEPS } from "@/config/training";
import { useWizardStore } from "@/stores/training";
import { ArrowLeft02Icon, ArrowRight02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { useShallow } from "zustand/react/shallow";

export function WizardFooter() {
  const { currentStep, prevStep, nextStep, canProceed } = useWizardStore(
    useShallow((s) => ({
      currentStep: s.currentStep,
      prevStep: s.prevStep,
      nextStep: s.nextStep,
      canProceed: s.canProceed(),
    })),
  );
  const navigate = useNavigate();
  const isFirst = currentStep === 1;
  const isLast = currentStep === STEPS.length;

  return (
    <footer>
      <div className="flex items-center justify-between p-6">
        <Button
          variant="outline"
          className="px-4 !pl-4"
          onClick={prevStep}
          disabled={isFirst}
        >
          <HugeiconsIcon icon={ArrowLeft02Icon} data-icon="inline-start" />
          Back
        </Button>
        {isLast ? (
          <Button
            onClick={() => navigate({ to: "/studio" })}
            disabled={!canProceed}
            className="px-4 !pr-4"
          >
            Go to Studio
            <HugeiconsIcon icon={ArrowRight02Icon} data-icon="inline-end" />
          </Button>
        ) : (
          <Button
            onClick={nextStep}
            className="px-4 !pl-4"
            disabled={!canProceed}
          >
            Continue
            <HugeiconsIcon icon={ArrowRight02Icon} data-icon="inline-end" />
          </Button>
        )}
      </div>
    </footer>
  );
}
