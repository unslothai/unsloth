import { Progress } from "@/components/ui/progress";
import { STEPS } from "@/config/training";
import { useWizardStore } from "@/stores/training";
import { WizardStepItem } from "./wizard-step-item";

export function WizardSidebar() {
  const currentStep = useWizardStore((s) => s.currentStep);
  const progress = ((currentStep - 1) / (STEPS.length - 1)) * 100;

  return (
    <aside className="w-64 flex flex-col gap-4 p-6 shrink-0 bg-muted/70">
      <div className="flex items-center gap-3 py-2">
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
      <Progress value={progress} className="h-1.5" />
      <nav className="flex flex-col gap-1">
        {STEPS.map((step) => (
          <WizardStepItem key={step.number} step={step} />
        ))}
      </nav>
    </aside>
  );
}
