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
        id: "nav",
        target: "navbar",
        title: "Quick orientation",
        body: "Studio is where you fine-tune. Export ships results. Chat is for poking at models. This tour is Studio-only (for now).",
      },
      {
        id: "local-model",
        target: "studio-local-model",
        title: "Local model path",
        body: "Point to a local folder (`./models/...`) or a custom HF repo. Use this when you already downloaded weights.",
      },
      {
        id: "base-model",
        target: "studio-base-model",
        title: "Base model from Hugging Face",
        body: "Search Hub here. Paste `org/model` too. Pick something close to your target domain to save compute.",
      },
      {
        id: "method",
        target: "studio-method",
        title: "Method: QLoRA vs LoRA vs Full",
        body: "QLoRA: lowest VRAM (4-bit). LoRA: fast + solid (16-bit adapters). Full: slowest, highest cost, updates all weights.",
      },
      {
        id: "dataset",
        target: "studio-dataset",
        title: "Dataset",
        body: "Search Hub or paste `user/dataset`. Preview a few rows before you burn hours of compute.",
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
