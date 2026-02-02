import { STEPS } from "@/config/training";
import { useWizardStore } from "@/stores/training";
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
  const currentStep = useWizardStore((s) => s.currentStep);
  const stepConfig = STEPS[currentStep - 1];
  const StepComponent = STEP_COMPONENTS[currentStep];
  const mascotSrc = STEP_MASCOTS[currentStep];

  return (
    <main className="flex-1 flex flex-col overflow-y-auto">
      <header className="flex items-center gap-4 p-6 pb-4">
        <img src={mascotSrc} alt="Unsloth mascot" className="size-14" />
        <div className="flex flex-col min-w-0">
          <h1 className="text-xl font-semibold">{stepConfig.title}</h1>
          <p className="text-sm text-muted-foreground">
            {stepConfig.description}
          </p>
        </div>
        <p className="ml-auto shrink-0 text-xs text-muted-foreground uppercase tracking-wider">
          Step {currentStep} of {STEPS.length}
        </p>
      </header>
      <div className="flex-1 p-6 pt-2">
        <StepComponent />
      </div>
    </main>
  );
}
