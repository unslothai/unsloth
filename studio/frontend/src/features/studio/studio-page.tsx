import { Button } from "@/components/ui/button";
import {
  shouldShowTrainingView,
  useTrainingActions,
  useTrainingRuntimeLifecycle,
  useTrainingRuntimeStore,
} from "@/features/training";
import { GuidedTour, type TourStep } from "@/features/tour";
import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useMemo, useState } from "react";
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
  const tourEnabled = hasHydratedRuntime && !isHydratingRuntime && !showTrainingView;
  const [tourOpen, setTourOpen] = useState(false);

  const tourSteps = useMemo<TourStep[]>(
    () => [
      {
        id: "header",
        target: "studio-header",
        title: "This page is a pipeline",
        body: "Pick a base model, attach data, tune params, then launch training. We’ll hit the 5 things that matter.",
      },
      {
        id: "model",
        target: "studio-model",
        title: "Choose model + method",
        body: "Base model sets your ceiling. Method sets speed vs quality. Start here, then everything else follows.",
      },
      {
        id: "dataset",
        target: "studio-dataset",
        title: "Bring a dataset",
        body: "Search Hub or paste `user/dataset`. Preview a few rows before you commit hours of compute.",
      },
      {
        id: "params",
        target: "studio-params",
        title: "Dial hyperparams",
        body: "Epochs + context length + LR. Keep it boring: small changes, one at a time.",
      },
      {
        id: "start",
        target: "studio-start",
        title: "Start training",
        body: "One click. If it fails, the error text is the first place to look (token, path, config).",
      },
      {
        id: "save",
        target: "studio-save",
        title: "Save config",
        body: "Save good runs. Repeatability beats vibe. You can iterate from a known baseline.",
      },
    ],
    [],
  );

  useEffect(() => {
    if (!tourEnabled) return;
    if (localStorage.getItem(STUDIO_TOUR_KEY)) return;
    setTourOpen(true);
  }, [tourEnabled]);

  return (
    <div className="min-h-screen bg-background">
      <main className="mx-auto max-w-7xl px-6 py-4">
        <GuidedTour
          open={tourOpen}
          onOpenChange={setTourOpen}
          steps={tourSteps}
          onSkip={() => localStorage.setItem(STUDIO_TOUR_KEY, "skipped")}
          onComplete={() => localStorage.setItem(STUDIO_TOUR_KEY, "done")}
        />

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
        <div data-tour="studio-header" className="mb-8 flex flex-col gap-0.5">
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
