// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  shouldShowTrainingView,
  useDatasetPreviewDialogStore,
  useTrainingConfigStore,
  useTrainingRuntimeLifecycle,
  useTrainingRuntimeStore,
} from "@/features/training";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { studioTourSteps, studioTrainingTourSteps } from "./tour";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useCallback, useEffect, useMemo, useState } from "react";
import { useSidebar } from "@/components/ui/sidebar";
import { DatasetPreviewDialog } from "./sections/dataset-preview-dialog";
import { DatasetSection } from "./sections/dataset-section";
import { ModelSection } from "./sections/model-section";
import { ParamsSection } from "./sections/params-section";
import { TrainingSection } from "./sections/training-section";
import { LiveTrainingView } from "./live-training-view";
import { HistoricalTrainingView } from "./historical-training-view";
import { HistoryCardGrid } from "./history-card-grid";

export function StudioPage(): ReactElement {
  useTrainingRuntimeLifecycle();
  const showTrainingView = useTrainingRuntimeStore(shouldShowTrainingView);
  const isTrainingRunning = useTrainingRuntimeStore((state) => state.isTrainingRunning);
  const currentJobId = useTrainingRuntimeStore((state) => state.jobId);
  const runtimeMessage = useTrainingRuntimeStore((state) => state.message);
  const isHydratingRuntime = useTrainingRuntimeStore((state) => state.isHydrating);
  const hasHydratedRuntime = useTrainingRuntimeStore((state) => state.hasHydrated);

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

  const [requestedTab, setRequestedTab] = useState("configure");
  const selectedHistoryRunId = useTrainingRuntimeStore((s) => s.selectedHistoryRunId);
  const setSelectedHistoryRunId = useTrainingRuntimeStore((s) => s.setSelectedHistoryRunId);

  useEffect(() => {
    return () => setSelectedHistoryRunId(null);
  }, [setSelectedHistoryRunId]);

  // Derive activeTab: auto-switch to "current-run" only while training is
  // genuinely running.  Once training ends, honour whatever tab the user clicks.
  // If requestedTab is "current-run" but there's nothing to show, fall back to "configure".
  const activeTab =
    isTrainingRunning && requestedTab !== "history"
      ? "current-run"
      : requestedTab === "current-run" && !showTrainingView
        ? "configure"
        : requestedTab;

  const { setPinned } = useSidebar();
  const pinSidebar = useCallback(() => setPinned(true), [setPinned]);

  const tourEnabled = hasHydratedRuntime && !isHydratingRuntime;
  const isConfigTour = activeTab === "configure";
  const baseTourSteps = activeTab === "current-run" ? studioTrainingTourSteps : studioTourSteps;
  // Inject onEnter for navbar-targeting steps so the sidebar expands during the tour.
  const tourSteps = useMemo(
    () =>
      baseTourSteps.map((step) =>
        step.target === "navbar" ? { ...step, onEnter: pinSidebar } : step,
      ),
    [baseTourSteps, pinSidebar],
  );
  const tour = useGuidedTourController({
    id: "studio",
    steps: tourSteps,
    enabled: tourEnabled,
  });

  const setTourOpen = tour.setOpen;
  useEffect(() => {
    setTourOpen(false);
  }, [activeTab, setTourOpen]);

  // When training auto-switches us to "current-run", persist that in
  // requestedTab so the user stays on results after training ends.
  useEffect(() => {
    if (isTrainingRunning && requestedTab !== "history" && requestedTab !== "current-run") {
      setRequestedTab("current-run");
      setSelectedHistoryRunId(null);
    }
  }, [isTrainingRunning, requestedTab]);

  // Selecting a run from the sidebar only sets selectedHistoryRunId; auto-switch
  // to the History tab so the main panel reflects the selection.
  useEffect(() => {
    if (selectedHistoryRunId && requestedTab !== "history") {
      setRequestedTab("history");
    }
  }, [selectedHistoryRunId, requestedTab]);

  useEffect(() => {
    ensureModelDefaultsLoaded();
    ensureDatasetChecked();
  }, [selectedModel, ensureModelDefaultsLoaded, ensureDatasetChecked]);

  function handleTabChange(value: string) {
    setRequestedTab(value);
    if (value !== "history") {
      setSelectedHistoryRunId(null);
    }
  }

  const subtitle = (() => {
    if (activeTab === "current-run") return runtimeMessage || "Training in progress";
    if (activeTab === "history")
      return selectedHistoryRunId ? "Viewing past run" : "View past training runs";
    return "Configure and start training";
  })();

  return (
    <div className="relative min-h-[calc(100dvh-var(--studio-titlebar-height,0px))] bg-background">
      <main className="relative z-10 mx-auto max-w-7xl px-4 py-8 sm:px-6">
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

        <div className="mb-6 flex flex-col gap-0.5 sm:mb-8">
          <h1 className="text-2xl font-semibold tracking-tight">
            Fine-tuning Studio
          </h1>
          <p className="text-sm text-muted-foreground">{subtitle}</p>
        </div>

        {!hasHydratedRuntime && isHydratingRuntime ? (
          <div className="rounded-xl border bg-card p-8 text-sm text-muted-foreground">
            Loading training runtime...
          </div>
        ) : (
          <Tabs value={activeTab} onValueChange={handleTabChange}>
            <div className="flex items-center gap-3">
              {selectedHistoryRunId && activeTab === "history" && (
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="rounded-full text-muted-foreground"
                  onClick={() => setSelectedHistoryRunId(null)}
                  aria-label="Back to history"
                >
                  <HugeiconsIcon icon={ArrowLeft01Icon} className="size-4" />
                </Button>
              )}
              <TabsList variant="line">
                <TabsTrigger value="configure" disabled={isTrainingRunning}>
                  Configure
                </TabsTrigger>
                <TabsTrigger value="current-run" disabled={!showTrainingView}>
                  Current Run
                </TabsTrigger>
                <TabsTrigger value="history">History</TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="configure">
              <div className="flex min-w-0 flex-col gap-4 md:gap-6">
                <ModelSection />
                <div className="grid min-w-0 grid-cols-1 items-start gap-4 md:grid-cols-2 md:gap-6 xl:grid-cols-3 xl:gap-6">
                  <DatasetSection />
                  <ParamsSection />
                  <TrainingSection />
                </div>
              </div>
            </TabsContent>

            <TabsContent value="current-run">
              <LiveTrainingView />
            </TabsContent>

            <TabsContent value="history">
              {selectedHistoryRunId ? (
                <HistoricalTrainingView runId={selectedHistoryRunId} />
              ) : (
                <HistoryCardGrid onSelectRun={(runId) => {
                  if (runId === currentJobId && isTrainingRunning) {
                    handleTabChange("current-run");
                  } else {
                    setSelectedHistoryRunId(runId);
                  }
                }} onResumeStarted={() => {
                  setSelectedHistoryRunId(null);
                  handleTabChange("current-run");
                }} />
              )}
            </TabsContent>
          </Tabs>
        )}
      </main>
    </div>
  );
}
