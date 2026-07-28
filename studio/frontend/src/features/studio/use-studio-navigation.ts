// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  shouldShowTrainingView,
  useTrainingRuntimeStore,
} from "@/features/training";
import type { useT } from "@/i18n";
import { useCallback, useEffect, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";

export type TrainSubTab = "configure" | "current-run" | "history";

function initialStudioTab(
  selectedHistoryRunId: string | null,
  isTrainingRunning: boolean,
): TrainSubTab {
  if (selectedHistoryRunId) {
    return "history";
  }
  return isTrainingRunning ? "current-run" : "configure";
}

function activeStudioTab(
  requestedTab: TrainSubTab,
  isTrainingRunning: boolean,
  showTrainingView: boolean,
): TrainSubTab {
  if (isTrainingRunning && requestedTab !== "history") {
    return "current-run";
  }
  if (requestedTab === "current-run" && !showTrainingView) {
    return "configure";
  }
  return requestedTab;
}

function runtimeRequestedTab({
  isTrainingRunning,
  previousIsTrainingRunning,
  previousSelectedHistoryRunId,
  requestedTab,
  selectedHistoryRunId,
}: {
  isTrainingRunning: boolean;
  previousIsTrainingRunning: boolean;
  previousSelectedHistoryRunId: string | null;
  requestedTab: TrainSubTab;
  selectedHistoryRunId: string | null;
}): TrainSubTab | null {
  if (
    selectedHistoryRunId &&
    selectedHistoryRunId !== previousSelectedHistoryRunId &&
    requestedTab !== "history"
  ) {
    return "history";
  }
  if (
    isTrainingRunning &&
    !previousIsTrainingRunning &&
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
    isTrainingRunning,
    selectedHistoryRunId,
    setCurrentRunViewActive,
    setSelectedHistoryRunId,
  } = useTrainingRuntimeStore(
    useShallow((state) => ({
      currentJobId: state.jobId,
      isTrainingRunning: state.isTrainingRunning,
      selectedHistoryRunId: state.selectedHistoryRunId,
      setCurrentRunViewActive: state.setCurrentRunViewActive,
      setSelectedHistoryRunId: state.setSelectedHistoryRunId,
    })),
  );
  const [requestedTab, setRequestedTabState] = useState<TrainSubTab>(() =>
    initialStudioTab(selectedHistoryRunId, isTrainingRunning),
  );
  const requestedTabRef = useRef(requestedTab);
  const setRequestedTab = useCallback((next: TrainSubTab) => {
    requestedTabRef.current = next;
    setRequestedTabState(next);
  }, []);
  const activeTab = activeStudioTab(
    requestedTab,
    isTrainingRunning,
    showTrainingView,
  );

  useEffect(() => {
    return () => setSelectedHistoryRunId(null);
  }, [setSelectedHistoryRunId]);

  useEffect(() => {
    setCurrentRunViewActive(activeTab === "current-run");
    return () => setCurrentRunViewActive(false);
  }, [activeTab, setCurrentRunViewActive]);

  useEffect(() => {
    return useTrainingRuntimeStore.subscribe((state, previousState) => {
      const nextTab = runtimeRequestedTab({
        isTrainingRunning: state.isTrainingRunning,
        previousIsTrainingRunning: previousState.isTrainingRunning,
        previousSelectedHistoryRunId: previousState.selectedHistoryRunId,
        requestedTab: requestedTabRef.current,
        selectedHistoryRunId: state.selectedHistoryRunId,
      });
      if (!nextTab) {
        return;
      }
      setRequestedTab(nextTab);
      if (nextTab === "current-run") {
        setSelectedHistoryRunId(null);
      }
    });
  }, [setRequestedTab, setSelectedHistoryRunId]);

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
      if (runId === currentJobId && isTrainingRunning) {
        handleTabChange("current-run");
        return;
      }
      setSelectedHistoryRunId(runId);
    },
    [currentJobId, handleTabChange, isTrainingRunning, setSelectedHistoryRunId],
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
    isTrainingRunning,
    selectedHistoryRunId,
    showTrainingView,
  };
}
