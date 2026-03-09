// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { Button } from "@/components/ui/button";
import {
  shouldShowTrainingView,
  useDatasetPreviewDialogStore,
  useTrainingActions,
  useTrainingConfigStore,
  useTrainingRuntimeLifecycle,
  useTrainingRuntimeStore,
} from "@/features/training";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { studioTourSteps, studioTrainingTourSteps } from "./tour";
import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect } from "react";
import { DatasetPreviewDialog } from "./sections/dataset-preview-dialog";
import { DatasetSection } from "./sections/dataset-section";
import { ModelSection } from "./sections/model-section";
import { ParamsSection } from "./sections/params-section";
import { TrainingSection } from "./sections/training-section";
import { TrainingView } from "./training-view";

const STUDIO_TOUR_KEY = "tour:studio:v1";

export function StudioPage(): ReactElement {
  useTrainingRuntimeLifecycle();
  const showTrainingView = useTrainingRuntimeStore(shouldShowTrainingView);
  const isTrainingRunning = useTrainingRuntimeStore((state) => state.isTrainingRunning);
  const runtimeMessage = useTrainingRuntimeStore((state) => state.message);
  const runtimePhase = useTrainingRuntimeStore((state) => state.phase);
  const isHydratingRuntime = useTrainingRuntimeStore((state) => state.isHydrating);
  const hasHydratedRuntime = useTrainingRuntimeStore((state) => state.hasHydrated);
  const { dismissTrainingRun } = useTrainingActions();

  const config = useTrainingConfigStore();
  const selectedModel = useTrainingConfigStore((s) => s.selectedModel);
  const ensureModelDefaultsLoaded = useTrainingConfigStore(
    (s) => s.ensureModelDefaultsLoaded,
  );
  const ensureDatasetChecked = useTrainingConfigStore(
    (s) => s.ensureDatasetChecked,
  );
  const dialogOpen = useDatasetPreviewDialogStore((s) => s.open);
  const dialogMode = useDatasetPreviewDialogStore((s) => s.mode);
  const dialogInitial = useDatasetPreviewDialogStore((s) => s.initialData);
  const closeDialog = useDatasetPreviewDialogStore((s) => s.close);

  const stopRequested = useTrainingRuntimeStore((state) => state.stopRequested);
  const canGoBack =
    showTrainingView &&
    !isHydratingRuntime &&
    (stopRequested ||
      (!isTrainingRunning &&
        (runtimePhase === "stopped" ||
          runtimePhase === "error" ||
          runtimePhase === "completed" ||
          runtimePhase === "idle")));
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

  const setTourOpen = tour.setOpen;
  useEffect(() => {
    setTourOpen(false);
  }, [showTrainingView, setTourOpen]);

  useEffect(() => {
    ensureModelDefaultsLoaded();
    ensureDatasetChecked();
  }, [selectedModel, ensureModelDefaultsLoaded, ensureDatasetChecked]);

  return (
    <div className="relative min-h-screen overflow-hidden bg-background">
      <main className="relative z-10 mx-auto max-w-7xl px-4 py-4 sm:px-6">
        <GuidedTour {...tour.tourProps} celebrate={isConfigTour} />

        <DatasetPreviewDialog
          open={dialogOpen}
          onOpenChange={(open) => {
            if (!open) closeDialog();
          }}
          datasetSource={config.datasetSource}
          datasetName={
            config.datasetSource === "huggingface" ? config.dataset : config.uploadedFile
          }
          hfToken={config.hfToken.trim() || null}
          datasetSubset={config.datasetSubset}
          datasetSplit={config.datasetSplit}
          mode={dialogMode}
          initialData={dialogInitial}
          isVlm={config.isVisionModel && config.isDatasetImage === true}
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

        <div className="mb-6 flex flex-col gap-0.5 sm:mb-8">
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
          <div className="grid grid-cols-1 items-start gap-4 md:grid-cols-2 md:gap-6 xl:grid-cols-12">
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
