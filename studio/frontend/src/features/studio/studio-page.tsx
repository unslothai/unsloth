

import { MediaPageLink } from "@/components/media-page-link";
import { usePlatformStore } from "@/config/env";
import { Button } from "@/components/ui/button";
import { useSidebar } from "@/components/ui/sidebar";
import { Spinner } from "@/components/ui/spinner";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { FEATURE_IMAGES } from "@/config/disabled-features";
import { useHfTokenStore } from "@/features/hub";
import { useImageWorkflowStore } from "@/features/images/stores/image-workflow-store";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import {
  useDatasetPreviewDialogStore,
  useTrainingConfigStore,
  useTrainingRuntimeLifecycle,
  useTrainingRuntimeStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { ArrowLeft01Icon, Image03Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useRef,
} from "react";
import { useShallow } from "zustand/react/shallow";
import { HistoricalTrainingView } from "./historical-training-view";
import { HistoryCardGrid } from "./history-card-grid";
import { useTrainingCacheReconciliation } from "./hooks/use-training-cache-reconciliation";
import { LiveTrainingView } from "./live-training-view";
import { DatasetPreviewDialog } from "./sections/dataset-preview-dialog";
import { TrainSubNav } from "./studio-navigation";
import { studioTourSteps, studioTrainingTourSteps } from "./tour";
import {
  type TrainSubTab,
  getStudioSubtitle,
  useStudioNavigation,
} from "./use-studio-navigation";
import { RunPreviewCard } from "./wizard/run-preview-card";
import { StartTrainingCta } from "./wizard/start-training-cta";
import { useParamMode } from "./wizard/training-param-mode";
import { TrainingWizard } from "./wizard/training-wizard";

export function StudioPage(): ReactElement {
  const t = useT();
  const [paramMode, setParamMode] = useParamMode();
  useTrainingRuntimeLifecycle();
  useTrainingCacheReconciliation();
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
      datasetKnownCached: s.datasetKnownCached,
      datasetLocalPath: s.datasetLocalPath,
      datasetStreaming: s.datasetStreaming,
      datasetSubset: s.datasetSubset,
      datasetSplit: s.datasetSplit,
      isVisionModel: s.isVisionModel,
      isDatasetImage: s.isDatasetImage,
    })),
  );
  // Unknown until /api/health reports; see the showTrainingHydrating note below.
  const capabilitiesUnknown = usePlatformStore((s) => s.capabilitiesUnknown());
  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const navigate = useNavigate();
  // Once the verdict lands and it really is chat-only, leave: the guard let this load on the
  // guess, so without this a chat-only host would sit on a Train page it cannot use.
  useEffect(() => {
    if (capabilitiesUnknown || !chatOnly) return;
    void navigate({ to: "/chat", replace: true });
  }, [capabilitiesUnknown, chatOnly, navigate]);
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

  const {
    activeTab,
    clearHistorySelection,
    handleHistoryRunSelected,
    handleResumeStarted,
    handleTabChange,
    trainingRunActive,
    selectedHistoryRunId,
    showTrainingView,
  } = useStudioNavigation();

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
    if (selectedModel) {
      ensureModelDefaultsLoaded();
    }
    ensureDatasetChecked();
  }, [selectedModel, ensureModelDefaultsLoaded, ensureDatasetChecked]);

  const subtitle = getStudioSubtitle({
    activeTab,
    runtimeMessage,
    selectedHistoryRunId,
    t,
  });

  // The root guard now lets /studio through while the hardware verdict is out (redirecting on
  // the browser-platform guess strands a healthy host on /chat), so the page owns the wait:
  // show the same loading panel as a hydrating runtime rather than a half-built wizard. The
  // wait ends by itself: AppSidebar (mounted on this route) re-reads /api/health while the
  // verdict is unknown and writes it to this same store, so no second poll is needed here.
  const showTrainingHydrating =
    capabilitiesUnknown || (!hasHydratedRuntime && isHydratingRuntime);
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
              <div className="flex min-w-0 flex-wrap items-center gap-3 border-b border-border/60">
                {showHistoryBack && (
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    className="-ml-1 rounded-full text-muted-foreground"
                    onClick={clearHistorySelection}
                    aria-label={t("studio.backToHistory")}
                  >
                    <HugeiconsIcon icon={ArrowLeft01Icon} className="size-4" />
                  </Button>
                )}
                <TrainSubNav
                  value={activeTab}
                  trainingRunActive={trainingRunActive}
                  showTrainingView={showTrainingView}
                />
                {/* Image training is a mode of the Images page, not a route, so it sits
                    beside the sub-nav rather than in it. */}
                {FEATURE_IMAGES && (
                  <div className="ml-auto flex items-center gap-2 pb-2">
                    <MediaPageLink
                      to="/images"
                      label={t("studio.imageTraining")}
                      tooltip={t("studio.goToImageTraining")}
                      icon={Image03Icon}
                      onNavigate={() =>
                        useImageWorkflowStore.getState().setPageMode("train")
                      }
                    />
                  </div>
                )}
              </div>
            )}
          </header>

          <div className="flex w-full flex-col gap-6">
            <GuidedTour {...tour.tourProps} celebrate={isConfigTour} />

            {showTrainingHydrating ? (
              <div className="flex items-center gap-3 rounded-2xl border border-border/60 p-8 text-sm text-muted-foreground">
                <Spinner className="size-4 shrink-0" />
                {t("studio.loadingRuntime")}
              </div>
            ) : (
              <>
                <TabsContent value="configure" className="mt-0">
                  <div className="@container/train-configure">
                    <div className="grid grid-cols-1 gap-8 @5xl/train-configure:grid-cols-[minmax(0,1fr)_320px] @5xl/train-configure:gap-10">
                      <div className="min-w-0">
                        <TrainingWizard
                          paramMode={paramMode}
                          onParamModeChange={setParamMode}
                        />
                      </div>
                      <div className="@5xl/train-configure:sticky @5xl/train-configure:top-6 @5xl/train-configure:self-start">
                        <RunPreviewCard
                          paramMode={paramMode}
                          startCta={<StartTrainingCta />}
                        />
                      </div>
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
                      onResumeStarted={handleResumeStarted}
                    />
                  ) : (
                    <HistoryCardGrid
                      onSelectRun={handleHistoryRunSelected}
                      onResumeStarted={handleResumeStarted}
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
            datasetKnownCached={config.datasetKnownCached}
            datasetLocalPath={config.datasetLocalPath}
            datasetStreaming={config.datasetStreaming}
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
