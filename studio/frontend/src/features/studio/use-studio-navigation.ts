// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  isTrainingRunActive,
  shouldShowTrainingView,
  useTrainingRuntimeStore,
} from "@/features/training";
import type { useT } from "@/i18n";
import { useCallback, useEffect, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";

export type TrainSubTab = "configure" | "current-run" | "history";

function initialStudioTab(
  selectedHistoryRunId: string | null,
  trainingRunActive: boolean,
): TrainSubTab {
  if (selectedHistoryRunId) {
    return "history";
  }
  return trainingRunActive ? "current-run" : "configure";
}

function activeStudioTab(
  requestedTab: TrainSubTab,
  trainingRunActive: boolean,
  showTrainingView: boolean,
): TrainSubTab {
  if (trainingRunActive && requestedTab !== "history") {
    return "current-run";
  }
  if (requestedTab === "current-run" && !showTrainingView) {
    return "configure";
  }
  return requestedTab;
}

function runtimeRequestedTab({
  jobId,
  previousTrainingRunActive,
  previousJobId,
  previousSelectedHistoryRunId,
  requestedTab,
  selectedHistoryRunId,
  trainingRunActive,
}: {
  jobId: string | null;
  previousTrainingRunActive: boolean;
  previousJobId: string | null;
  previousSelectedHistoryRunId: string | null;
  requestedTab: TrainSubTab;
  selectedHistoryRunId: string | null;
  trainingRunActive: boolean;
}): TrainSubTab | null {
  if (
    selectedHistoryRunId &&
    selectedHistoryRunId !== previousSelectedHistoryRunId &&
    requestedTab !== "history"
  ) {
    return "history";
  }
  if (
    ((jobId !== null && jobId !== previousJobId && trainingRunActive) ||
      (trainingRunActive && !previousTrainingRunActive)) &&
    requestedTab !== "history" &&
    requestedTab !== "current-run"
  ) {
    return "current-run";
  }
  return null;
}

export function getStudioSubtitle({
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

export function useStudioNavigation() {
  const showTrainingView = useTrainingRuntimeStore(shouldShowTrainingView);
  const {
    currentJobId,
    trainingRunActive,
    selectedHistoryRunId,
    setCurrentRunViewActive,
    setSelectedHistoryRunId,
  } = useTrainingRuntimeStore(
    useShallow((state) => ({
      currentJobId: state.jobId,
      trainingRunActive: isTrainingRunActive(state),
      selectedHistoryRunId: state.selectedHistoryRunId,
      setCurrentRunViewActive: state.setCurrentRunViewActive,
      setSelectedHistoryRunId: state.setSelectedHistoryRunId,
    })),
  );
  const [requestedTab, setRequestedTabState] = useState<TrainSubTab>(() =>
    initialStudioTab(selectedHistoryRunId, trainingRunActive),
  );
  const requestedTabRef = useRef(requestedTab);
  const setRequestedTab = useCallback((next: TrainSubTab) => {
    requestedTabRef.current = next;
    setRequestedTabState(next);
  }, []);
  const activeTab = activeStudioTab(
    requestedTab,
    trainingRunActive,
    showTrainingView,
  );

  useEffect(() => {
    setCurrentRunViewActive(activeTab === "current-run");
    return () => setCurrentRunViewActive(false);
  }, [activeTab, setCurrentRunViewActive]);

  useEffect(() => {
    if (activeTab !== "current-run" || selectedHistoryRunId === null) {
      return;
    }
    if (
      requestedTabRef.current === "current-run" &&
      useTrainingRuntimeStore.getState().selectedHistoryRunId ===
        selectedHistoryRunId
    ) {
      setSelectedHistoryRunId(null);
    }
  }, [activeTab, selectedHistoryRunId, setSelectedHistoryRunId]);

  useEffect(() => {
    return useTrainingRuntimeStore.subscribe((state, previousState) => {
      const nextTab = runtimeRequestedTab({
        jobId: state.jobId,
        previousJobId: previousState.jobId,
        previousTrainingRunActive: isTrainingRunActive(previousState),
        previousSelectedHistoryRunId: previousState.selectedHistoryRunId,
        requestedTab: requestedTabRef.current,
        selectedHistoryRunId: state.selectedHistoryRunId,
        trainingRunActive: isTrainingRunActive(state),
      });
      if (!nextTab) {
        return;
      }
      setRequestedTab(nextTab);
    });
  }, [setRequestedTab]);

  const handleTabChange = useCallback(
    (value: TrainSubTab) => {
      setRequestedTab(value);
      if (value !== "history") {
        setSelectedHistoryRunId(null);
      }
    },
    [setRequestedTab, setSelectedHistoryRunId],
  );

  const clearHistorySelection = useCallback(() => {
    setSelectedHistoryRunId(null);
  }, [setSelectedHistoryRunId]);

  const handleHistoryRunSelected = useCallback(
    (runId: string) => {
      if (runId === currentJobId && trainingRunActive) {
        handleTabChange("current-run");
        return;
      }
      setSelectedHistoryRunId(runId);
    },
    [currentJobId, handleTabChange, setSelectedHistoryRunId, trainingRunActive],
  );

  const handleResumeStarted = useCallback(() => {
    setSelectedHistoryRunId(null);
    handleTabChange("current-run");
  }, [handleTabChange, setSelectedHistoryRunId]);

  return {
    activeTab,
    clearHistorySelection,
    handleHistoryRunSelected,
    handleResumeStarted,
    handleTabChange,
    trainingRunActive,
    selectedHistoryRunId,
    showTrainingView,
  };
}
