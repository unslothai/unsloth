import { useWizardStore } from "@/stores/training";
import type { ReactElement } from "react";
import { DatasetSection } from "./sections/dataset-section";
import { ModelSection } from "./sections/model-section";
import { ParamsSection } from "./sections/params-section";
import { TrainingSection } from "./sections/training-section";
import { TrainingView } from "./training-view";

export function StudioPage(): ReactElement {
  const isTraining = useWizardStore((s) => s.isTraining);

  return (
    <div className="min-h-screen bg-background">
      <main className="mx-auto max-w-7xl px-6 py-8">
        {/* Header */}
        <div className="mb-8 flex flex-col gap-1">
          <h1 className="text-2xl font-semibold tracking-tight">
            Fine-tuning Studio
          </h1>
          <p className="text-sm text-muted-foreground">
            {isTraining
              ? "Training in progress"
              : "Configure and start training"}
          </p>
        </div>

        {isTraining ? (
          <TrainingView />
        ) : (
          <div className="grid grid-cols-12 items-start gap-6">
            <ModelSection />
            <DatasetSection />
            <ParamsSection />
            <TrainingSection />
          </div>
        )}
      </main>
    </div>
  );
}
