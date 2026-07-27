// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useSidebar } from "@/components/ui/sidebar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useHfTokenStore } from "@/features/hub";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import {
  shouldShowTrainingView,
  useDatasetPreviewDialogStore,
  useTrainingConfigStore,
  useTrainingRuntimeLifecycle,
  useTrainingRuntimeStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
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
import { useShallow } from "zustand/react/shallow";
import { HistoricalTrainingView } from "./historical-training-view";
import { HistoryCardGrid } from "./history-card-grid";
import { useTrainingCacheReconciliation } from "./hooks/use-training-cache-reconciliation";
import { LiveTrainingView } from "./live-training-view";
import { DatasetPreviewDialog } from "./sections/dataset-preview-dialog";
import { studioTourSteps, studioTrainingTourSteps } from "./tour";
import { RunPreviewCard } from "./wizard/run-preview-card";
import { StartTrainingCta, TrainingWizard } from "./wizard/training-wizard";

type TrainSubTab = "configure" | "current-run" | "history";

function getStudioSubtitle({
  activeTab,
  runtimeMessage,
  selectedHistoryRunId,
  t,
}: {
  activeTab: TrainSubTab;
  runtimeMessage: string;
  selectedHistoryRunId: string | null;
  t: ReturnType<typeof useT>;
}): string {
  if (activeTab === "current-run") {
    return runtimeMessage || t("studio.subtitles.trainingInProgress");
  }
  if (activeTab === "history") {
    return selectedHistoryRunId
      ? t("studio.subtitles.viewingPastRun")
      : t("studio.subtitles.viewPastRuns");
  }
  return t("studio.subtitles.configure");
}

function TrainSubNav({
  value,
  isTrainingRunning,
  showTrainingView,
}: {
  value: TrainSubTab;
  isTrainingRunning: boolean;
  showTrainingView: boolean;
}): ReactElement {
  const t = useT();
  const items: ReadonlyArray<{
    value: TrainSubTab;
    label: string;
    disabled: boolean;
  }> = [
    {
      value: "configure",
      label: t("studio.tabs.configure"),
      disabled: isTrainingRunning,
    },
    {
      value: "current-run",
      label: t("studio.tabs.currentRun"),
      disabled: !showTrainingView,
    },
    { value: "history", label: t("studio.tabs.history"), disabled: false },
  ];
  return (
    <TabsList
      unstyled={true}
      className="flex items-center gap-6 text-ui-13 tracking-nav"
    >
      {items.map((item) => {
        const active = value === item.value;
        return (
          <TabsTrigger
            key={item.value}
            value={item.value}
            disabled={item.disabled}
            indicatorClassName="hidden"
            className={cn(
              "relative h-9 flex-none select-none rounded-none border-0 px-0 py-0 text-ui-13 transition-colors disabled:cursor-not-allowed disabled:opacity-40",
              "after:pointer-events-none after:absolute after:inset-x-0 after:bottom-[-1px] after:h-[2px] after:rounded-full after:bg-foreground after:transition-opacity",
              active
                ? "font-semibold text-foreground after:opacity-100"
                : "text-muted-foreground hover:text-foreground after:opacity-0",
            )}
          >
            {item.label}
          </TabsTrigger>
        );
      })}
    </TabsList>
  );
}

export function StudioPage(): ReactElement {
  const t = useT();
  useTrainingRuntimeLifecycle();
  useTrainingCacheReconciliation();
  const showTrainingView = useTrainingRuntimeStore(shouldShowTrainingView);
  const isTrainingRunning = useTrainingRuntimeStore(
    (state) => state.isTrainingRunning,
  );
  const currentJobId = useTrainingRuntimeStore((state) => state.jobId);
  const runtimeMessage = useTrainingRuntimeStore((state) => state.message);
  const isHydratingRuntime = useTrainingRuntimeStore(
    (state) => state.isHydrating,
  );
  const hasHydratedRuntime = useTrainingRuntimeStore(
    (state) => state.hasHydrated,
  );

  const config = useTrainingConfigStore(
    useShallow((s) => ({
      datasetSource: s.datasetSource,
      dataset: s.dataset,
      uploadedFile: s.uploadedFile,
      datasetSubset: s.datasetSubset,
      datasetSplit: s.datasetSplit,
      isVisionModel: s.isVisionModel,
      isDatasetImage: s.isDatasetImage,
    })),
  );
  const hfToken = useHfTokenStore((s) => s.token);
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

  const selectedHistoryRunId = useTrainingRuntimeStore(
    (s) => s.selectedHistoryRunId,
  );
  const setSelectedHistoryRunId = useTrainingRuntimeStore(
    (s) => s.setSelectedHistoryRunId,
  );

  const setCurrentRunViewActive = useTrainingRuntimeStore(
    (s) => s.setCurrentRunViewActive,
  );
  const [requestedTab, setRequestedTabState] = useState<TrainSubTab>(() =>
    selectedHistoryRunId
      ? "history"
      : isTrainingRunning
        ? "current-run"
        : "configure",
  );
  const requestedTabRef = useRef(requestedTab);
  const setRequestedTab = useCallback((next: TrainSubTab) => {
    requestedTabRef.current = next;
    setRequestedTabState(next);
  }, []);

  useEffect(() => {
    return () => setSelectedHistoryRunId(null);
  }, [setSelectedHistoryRunId]);

  const activeTab: TrainSubTab =
    isTrainingRunning && requestedTab !== "history"
      ? "current-run"
      : requestedTab === "current-run" && !showTrainingView
        ? "configure"
        : requestedTab;

  useEffect(() => {
    setCurrentRunViewActive(activeTab === "current-run");
    return () => setCurrentRunViewActive(false);
  }, [activeTab, setCurrentRunViewActive]);

  const { setPinned } = useSidebar();
  const pinSidebar = useCallback(() => setPinned(true), [setPinned]);

  const tourEnabled = hasHydratedRuntime && !isHydratingRuntime;
  const isConfigTour = activeTab === "configure";
  const baseTourSteps =
    activeTab === "current-run" ? studioTrainingTourSteps : studioTourSteps;
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
  const previousTourTabRef = useRef(activeTab);
  useEffect(() => {
    if (previousTourTabRef.current === activeTab) {
      return;
    }
    previousTourTabRef.current = activeTab;
    setTourOpen(false);
  }, [activeTab, setTourOpen]);

  useEffect(() => {
    return useTrainingRuntimeStore.subscribe((state, previousState) => {
      if (
        state.selectedHistoryRunId &&
        state.selectedHistoryRunId !== previousState.selectedHistoryRunId &&
        requestedTabRef.current !== "history"
      ) {
        setRequestedTab("history");
        return;
      }
      if (
        state.isTrainingRunning &&
        !previousState.isTrainingRunning &&
        requestedTabRef.current !== "history" &&
        requestedTabRef.current !== "current-run"
      ) {
        setRequestedTab("current-run");
        setSelectedHistoryRunId(null);
      }
    });
  }, [setRequestedTab, setSelectedHistoryRunId]);

  useEffect(() => {
    if (selectedModel) {
      ensureModelDefaultsLoaded();
    }
    ensureDatasetChecked();
  }, [selectedModel, ensureModelDefaultsLoaded, ensureDatasetChecked]);

  function handleTabChange(value: TrainSubTab) {
    setRequestedTab(value);
    if (value !== "history") {
      setSelectedHistoryRunId(null);
    }
  }

  const subtitle = getStudioSubtitle({
    activeTab,
    runtimeMessage,
    selectedHistoryRunId,
    t,
  });

  const showTrainingHydrating = !hasHydratedRuntime && isHydratingRuntime;
  const showHistoryBack = activeTab === "history" && !!selectedHistoryRunId;

  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <Tabs
        value={activeTab}
        onValueChange={(value) => handleTabChange(value as TrainSubTab)}
        className="contents"
      >
        <div className="mx-auto flex w-full max-w-[1180px] flex-col gap-7 px-5 pb-20 pt-8 sm:px-9 sm:pt-10">
          <header className="font-heading flex flex-col gap-5">
            <div className="flex flex-col gap-0.5">
              <h1 className="page-title-halo text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
                {t("studio.routeTitle")}
              </h1>
              <p className="page-title-halo text-sm text-muted-foreground">
                {subtitle}
              </p>
            </div>
            {!showTrainingHydrating && (
              <div className="flex items-center gap-3 border-b border-border/60">
                {showHistoryBack && (
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    className="-ml-1 rounded-full text-muted-foreground"
                    onClick={() => setSelectedHistoryRunId(null)}
                    aria-label={t("studio.backToHistory")}
                  >
                    <HugeiconsIcon icon={ArrowLeft01Icon} className="size-4" />
                  </Button>
                )}
                <TrainSubNav
                  value={activeTab}
                  isTrainingRunning={isTrainingRunning}
                  showTrainingView={showTrainingView}
                />
              </div>
            )}
          </header>

          <div className="flex w-full flex-col gap-6">
            <GuidedTour {...tour.tourProps} celebrate={isConfigTour} />

            {showTrainingHydrating ? (
              <div className="rounded-2xl border border-border/60 p-8 text-sm text-muted-foreground">
                {t("studio.loadingRuntime")}
              </div>
            ) : (
              <>
                <TabsContent value="configure" className="mt-0">
                  <div className="grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,1fr)_320px] lg:gap-10">
                    <div className="min-w-0">
                      <TrainingWizard />
                    </div>
                    <div className="lg:sticky lg:top-6 lg:self-start">
                      <RunPreviewCard startCta={<StartTrainingCta />} />
                    </div>
                  </div>
                </TabsContent>
                <TabsContent value="current-run" className="mt-0">
                  <LiveTrainingView />
                </TabsContent>
                <TabsContent value="history" className="mt-0">
                  {selectedHistoryRunId ? (
                    <HistoricalTrainingView
                      runId={selectedHistoryRunId}
                      onResumeStarted={() => {
                        setSelectedHistoryRunId(null);
                        handleTabChange("current-run");
                      }}
                    />
                  ) : (
                    <HistoryCardGrid
                      onSelectRun={(runId) => {
                        if (runId === currentJobId && isTrainingRunning) {
                          handleTabChange("current-run");
                        } else {
                          setSelectedHistoryRunId(runId);
                        }
                      }}
                      onResumeStarted={() => {
                        setSelectedHistoryRunId(null);
                        handleTabChange("current-run");
                      }}
                    />
                  )}
                </TabsContent>
              </>
            )}
          </div>

          <DatasetPreviewDialog
            open={dialogOpen}
            onOpenChange={(open) => {
              if (!open) {
                closeDialog();
              }
            }}
            datasetSource={config.datasetSource}
            datasetName={
              config.datasetSource === "huggingface"
                ? config.dataset
                : config.uploadedFile
            }
            hfToken={hfToken.trim() || null}
            datasetSubset={config.datasetSubset}
            datasetSplit={config.datasetSplit}
            mode={dialogMode}
            initialData={dialogInitial}
            isVlm={config.isVisionModel && config.isDatasetImage === true}
          />
        </div>
      </Tabs>
    </div>
  );
}
