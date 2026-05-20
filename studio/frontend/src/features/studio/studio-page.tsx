// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PageHeading } from "@/components/layout";
import { SectionCardFlatContext } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import { useSidebar } from "@/components/ui/sidebar";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import {
  shouldShowTrainingView,
  useDatasetPreviewDialogStore,
  useTrainingConfigStore,
  useTrainingRuntimeLifecycle,
  useTrainingRuntimeStore,
} from "@/features/training";
import { cn } from "@/lib/utils";
import { useHfTokenStore } from "@/stores/hf-token-store";
import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { useShallow } from "zustand/react/shallow";
import { HistoricalTrainingView } from "./historical-training-view";
import { HistoryCardGrid } from "./history-card-grid";
import { RunPreviewCard } from "./run-preview-card";
import { StartTrainingCta, TrainingWizard } from "./training-wizard";
import { LiveTrainingView } from "./live-training-view";
import { DatasetPreviewDialog } from "./sections/dataset-preview-dialog";
import { studioTourSteps, studioTrainingTourSteps } from "./tour";

type TrainSubTab = "configure" | "current-run" | "history";

function TrainSubNav({
  value,
  onChange,
  isTrainingRunning,
  showTrainingView,
}: {
  value: TrainSubTab;
  onChange: (next: TrainSubTab) => void;
  isTrainingRunning: boolean;
  showTrainingView: boolean;
}): ReactElement {
  const items: ReadonlyArray<{
    value: TrainSubTab;
    label: string;
    disabled: boolean;
  }> = [
    { value: "configure", label: "Configure", disabled: isTrainingRunning },
    {
      value: "current-run",
      label: "Current Run",
      disabled: !showTrainingView,
    },
    { value: "history", label: "History", disabled: false },
  ];
  return (
    <div role="tablist" className="flex items-center gap-6 text-[13px] tracking-nav">
      {items.map((item) => {
        const active = value === item.value;
        return (
          <button
            key={item.value}
            role="tab"
            type="button"
            aria-selected={active}
            disabled={item.disabled}
            onClick={() => onChange(item.value)}
            className={cn(
              "relative h-9 select-none transition-colors disabled:cursor-not-allowed disabled:opacity-40",
              "after:pointer-events-none after:absolute after:inset-x-0 after:bottom-[-1px] after:h-[2px] after:rounded-full after:bg-foreground after:transition-opacity",
              active
                ? "font-semibold text-foreground after:opacity-100"
                : "text-muted-foreground hover:text-foreground after:opacity-0",
            )}
          >
            {item.label}
          </button>
        );
      })}
    </div>
  );
}

export function StudioPage(): ReactElement {
  useTrainingRuntimeLifecycle();
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

  const [requestedTab, setRequestedTab] = useState<TrainSubTab>("configure");
  const selectedHistoryRunId = useTrainingRuntimeStore(
    (s) => s.selectedHistoryRunId,
  );
  const setSelectedHistoryRunId = useTrainingRuntimeStore(
    (s) => s.setSelectedHistoryRunId,
  );

  useEffect(() => {
    return () => setSelectedHistoryRunId(null);
  }, [setSelectedHistoryRunId]);

  // Derive activeTab: auto-switch to "current-run" only while training is
  // genuinely running. Once training ends, honour whatever tab the user clicks.
  // If requestedTab is "current-run" but there's nothing to show, fall back to "configure".
  const activeTab: TrainSubTab =
    isTrainingRunning && requestedTab !== "history"
      ? "current-run"
      : requestedTab === "current-run" && !showTrainingView
        ? "configure"
        : requestedTab;

  const { setPinned } = useSidebar();
  const pinSidebar = useCallback(() => setPinned(true), [setPinned]);

  const tourEnabled = hasHydratedRuntime && !isHydratingRuntime;
  const isConfigTour = activeTab === "configure";
  const baseTourSteps =
    activeTab === "current-run" ? studioTrainingTourSteps : studioTourSteps;
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
  // biome-ignore lint/correctness/useExhaustiveDependencies: close the tour whenever the active sub-tab changes
  useEffect(() => {
    setTourOpen(false);
  }, [activeTab, setTourOpen]);

  // When training auto-switches us to "current-run", persist that in
  // requestedTab so the user stays on results after training ends.
  useEffect(() => {
    if (
      isTrainingRunning &&
      requestedTab !== "history" &&
      requestedTab !== "current-run"
    ) {
      setRequestedTab("current-run");
      setSelectedHistoryRunId(null);
    }
  }, [isTrainingRunning, requestedTab, setSelectedHistoryRunId]);

  // Selecting a run from the sidebar only sets selectedHistoryRunId; auto-switch
  // to the History tab so the main panel reflects the selection.
  useEffect(() => {
    if (selectedHistoryRunId && requestedTab !== "history") {
      setRequestedTab("history");
    }
  }, [selectedHistoryRunId, requestedTab]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: re-check defaults whenever the selected model changes
  useEffect(() => {
    ensureModelDefaultsLoaded();
    ensureDatasetChecked();
  }, [selectedModel, ensureModelDefaultsLoaded, ensureDatasetChecked]);

  const handleTrainSubTabChange = useCallback(
    (value: TrainSubTab) => {
      setRequestedTab(value);
      if (value !== "history") {
        setSelectedHistoryRunId(null);
      }
    },
    [setSelectedHistoryRunId],
  );

  const subtitle = (() => {
    if (activeTab === "current-run")
      return runtimeMessage || "Training in progress";
    if (activeTab === "history")
      return selectedHistoryRunId
        ? "Viewing past run"
        : "View past training runs";
    return "Configure and start a training run.";
  })();

  const showTrainingHydrating = !hasHydratedRuntime && isHydratingRuntime;

  const showHistoryBack =
    activeTab === "history" && !!selectedHistoryRunId;

  return (
    <div className="hub-page flex h-full min-h-0 flex-col bg-background">
      <div className="mx-auto flex w-full max-w-[1180px] flex-col gap-7 px-5 pb-20 pt-8 sm:px-9 sm:pt-10">
        <header className="font-heading flex flex-col gap-5">
          <PageHeading title="Train" subtitle="Configure and run training jobs." />
          {!showTrainingHydrating && (
            <div className="flex items-center gap-3 border-b border-border/60">
              {showHistoryBack && (
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="-ml-1 rounded-full text-muted-foreground"
                  onClick={() => setSelectedHistoryRunId(null)}
                  aria-label="Back to history"
                >
                  <HugeiconsIcon icon={ArrowLeft01Icon} className="size-4" />
                </Button>
              )}
              <TrainSubNav
                value={activeTab}
                onChange={handleTrainSubTabChange}
                isTrainingRunning={isTrainingRunning}
                showTrainingView={showTrainingView}
              />
              {activeTab !== "configure" && (
                <span className="ml-auto truncate text-[12px] text-muted-foreground">
                  {subtitle}
                </span>
              )}
            </div>
          )}
        </header>

        <div className="font-heading flex w-full flex-col gap-6">
            <GuidedTour {...tour.tourProps} celebrate={isConfigTour} />

            {showTrainingHydrating ? (
              <div className="rounded-2xl border border-border/60 p-8 text-sm text-muted-foreground">
                Loading training runtime...
              </div>
            ) : activeTab === "configure" ? (
              <SectionCardFlatContext.Provider value={true}>
                <div className="grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,1fr)_320px] lg:gap-10">
                  <div className="min-w-0">
                    <TrainingWizard />
                  </div>
                  <div className="lg:sticky lg:top-6 lg:self-start">
                    <RunPreviewCard startCta={<StartTrainingCta />} />
                  </div>
                </div>
              </SectionCardFlatContext.Provider>
            ) : activeTab === "current-run" ? (
              <LiveTrainingView />
            ) : selectedHistoryRunId ? (
              <HistoricalTrainingView runId={selectedHistoryRunId} />
            ) : (
              <HistoryCardGrid
                onSelectRun={(runId) => {
                  if (runId === currentJobId && isTrainingRunning) {
                    handleTrainSubTabChange("current-run");
                  } else {
                    setSelectedHistoryRunId(runId);
                  }
                }}
                onResumeStarted={() => {
                  setSelectedHistoryRunId(null);
                  handleTrainSubTabChange("current-run");
                }}
              />
            )}
        </div>

        <DatasetPreviewDialog
          open={dialogOpen}
          onOpenChange={(open) => {
            if (!open) closeDialog();
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
    </div>
  );
}
