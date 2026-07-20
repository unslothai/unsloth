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
import {
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useSidebar } from "@/components/ui/sidebar";
import { DatasetPreviewDialog } from "./sections/dataset-preview-dialog";
import { DatasetSection } from "./sections/dataset-section";
import { ModelSection } from "./sections/model-section";
import { ParamsSection } from "./sections/params-section";
import { TrainingSection } from "./sections/training-section";
import { TrainingQueuePanel } from "./sections/training-queue-panel";
import { QueueResumeBanner } from "./queue-resume-banner";
import { LiveTrainingView } from "./live-training-view";
import { HistoricalTrainingView } from "./historical-training-view";
import { HistoryCardGrid } from "./history-card-grid";
import { useT } from "@/i18n";

export function StudioPage(): ReactElement {
  const t = useT();
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
  const dialogStartIntent = useDatasetPreviewDialogStore((s) => s.startIntent);
  const closeDialog = useDatasetPreviewDialogStore((s) => s.close);

  const [requestedTab, setRequestedTab] = useState("configure");
  const selectedHistoryRunId = useTrainingRuntimeStore((s) => s.selectedHistoryRunId);
  const setSelectedHistoryRunId = useTrainingRuntimeStore((s) => s.setSelectedHistoryRunId);

  const setCurrentRunViewActive = useTrainingRuntimeStore(
    (s) => s.setCurrentRunViewActive,
  );

  useEffect(() => {
    return () => setSelectedHistoryRunId(null);
  }, [setSelectedHistoryRunId]);

  // Honour the user's clicked tab; the run-id effect below switches to
  // "current-run" once per run. If "current-run" has nothing to show,
  // use "configure".
  const activeTab =
    requestedTab === "current-run" && !showTrainingView
      ? "configure"
      : requestedTab;

  // Mirror "Current Run" tab state into the store so the sidebar can highlight
  // the run this view refers to. Cleared on unmount (leaving the studio page).
  useEffect(() => {
    setCurrentRunViewActive(activeTab === "current-run");
    return () => setCurrentRunViewActive(false);
  }, [activeTab, setCurrentRunViewActive]);

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

  // Key this to a job id change so back-to-back queued runs each switch once,
  // while the user can still navigate back to Configure during a given run.
  const lastStartedRunJobId = useRef<string | null>(null);
  useEffect(() => {
    const startedRunning =
      isTrainingRunning &&
      currentJobId !== null &&
      currentJobId !== lastStartedRunJobId.current;
    if (startedRunning) {
      lastStartedRunJobId.current = currentJobId;
    }
    if (startedRunning && requestedTab !== "history" && requestedTab !== "current-run") {
      setRequestedTab("current-run");
      setSelectedHistoryRunId(null);
    }
  }, [currentJobId, isTrainingRunning, requestedTab, setSelectedHistoryRunId]);

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
    if (activeTab === "current-run")
      return runtimeMessage || t("studio.subtitles.trainingInProgress");
    if (activeTab === "history")
      return selectedHistoryRunId
        ? t("studio.subtitles.viewingPastRun")
        : t("studio.subtitles.viewPastRuns");
    return t("studio.subtitles.configure");
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
          startIntent={dialogStartIntent}
          isVlm={config.isVisionModel && config.isDatasetImage === true}
        />

        <div className="mb-6 flex flex-col gap-0.5 sm:mb-8">
          <h1 className="text-[30px] font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-[34px]">
            {t("studio.title")}
          </h1>
          <p className="text-sm text-muted-foreground">{subtitle}</p>
        </div>

        {!hasHydratedRuntime && isHydratingRuntime ? (
          <div className="rounded-xl border bg-card p-8 text-sm text-muted-foreground">
            {t("studio.loadingRuntime")}
          </div>
        ) : (
          <>
          <QueueResumeBanner />
          <Tabs value={activeTab} onValueChange={handleTabChange}>
            <div className="flex items-center gap-3 pb-3">
              {selectedHistoryRunId && activeTab === "history" && (
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="rounded-full text-muted-foreground"
                  onClick={() => setSelectedHistoryRunId(null)}
                  aria-label={t("studio.backToHistory")}
                >
                  <HugeiconsIcon icon={ArrowLeft01Icon} className="size-4" />
                </Button>
              )}
              <TabsList variant="line">
                <TabsTrigger value="configure">
                  {t("studio.tabs.configure")}
                </TabsTrigger>
                <TabsTrigger value="current-run" disabled={!showTrainingView}>
                  {t("studio.tabs.currentRun")}
                </TabsTrigger>
                <TabsTrigger value="history">{t("studio.tabs.history")}</TabsTrigger>
              </TabsList>
              <div className="ml-auto">
                <TrainingQueuePanel />
              </div>
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
          </>
        )}
      </main>
    </div>
  );
}
