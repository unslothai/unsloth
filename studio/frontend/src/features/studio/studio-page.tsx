import { Button } from "@/components/ui/button";
import {
  shouldShowTrainingView,
  useTrainingActions,
  useTrainingRuntimeLifecycle,
  useTrainingRuntimeStore,
} from "@/features/training";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { studioTourSteps, studioTrainingTourSteps } from "@/features/studio/tour";
import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect } from "react";
import { DatasetSection } from "./sections/dataset-section";
import { ModelSection } from "./sections/model-section";
import { ParamsSection } from "./sections/params-section";
import { TrainingSection } from "./sections/training-section";
import { TrainingView } from "./training-view";

const STUDIO_TOUR_KEY = "tour:studio:v1";

export function StudioPage(): ReactElement {
  useTrainingRuntimeLifecycle();
  const showTrainingView = useTrainingRuntimeStore(shouldShowTrainingView);
  const runtimeMessage = useTrainingRuntimeStore((state) => state.message);
  const runtimePhase = useTrainingRuntimeStore((state) => state.phase);
  const isHydratingRuntime = useTrainingRuntimeStore((state) => state.isHydrating);
  const hasHydratedRuntime = useTrainingRuntimeStore((state) => state.hasHydrated);
  const { dismissTrainingRun } = useTrainingActions();

  const canGoBack = runtimePhase === "stopped" || runtimePhase === "error";
  const tourEnabled = hasHydratedRuntime && !isHydratingRuntime;
  const isConfigTour = !showTrainingView;
  const tourSteps = showTrainingView ? studioTrainingTourSteps : studioTourSteps;
  const tour = useGuidedTourController({
    id: "studio",
    steps: tourSteps,
    enabled: tourEnabled,
    autoKey: isConfigTour ? STUDIO_TOUR_KEY : undefined,
    autoWhen: isConfigTour,
  });

  useEffect(() => {
    tour.setOpen(false);
  }, [showTrainingView, tour.setOpen]);

  return (
    <div className="min-h-screen bg-background">
      <main className="mx-auto max-w-7xl px-6 py-4">
        <GuidedTour {...tour.tourProps} />

        {canGoBack && (
          <Button
            variant="ghost"
            size="sm"
            className="mb-2 cursor-pointer gap-1.5 text-muted-foreground"
            onClick={() => void dismissTrainingRun()}
          >
            <HugeiconsIcon icon={ArrowLeft01Icon} className="size-4" />
            Back to configuration
          </Button>
        )}

        {/* Header */}
        <div className="mb-8 flex flex-col gap-0.5">
          <h1 className="text-2xl font-semibold tracking-tight">
            Fine-tuning Studio
          </h1>
          <p className="text-sm text-muted-foreground">
            {showTrainingView
              ? runtimeMessage || "Training in progress"
              : "Configure and start training"}
          </p>
        </div>

        {!hasHydratedRuntime && isHydratingRuntime ? (
          <div className="rounded-xl border bg-card p-8 text-sm text-muted-foreground">
            Loading training runtime...
          </div>
        ) : showTrainingView ? (
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
