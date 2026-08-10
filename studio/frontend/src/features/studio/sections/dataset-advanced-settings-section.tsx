// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import { useTrainingConfigStore } from "@/features/training";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useEffect } from "react";
import { useShallow } from "zustand/react/shallow";
import { DatasetAdvancedSettings } from "./dataset-advanced-settings";
import { getDatasetStreamingBlockers } from "./dataset-panel-helpers";

export function DatasetAdvancedSettingsSection() {
  const t = useT();
  const platformDeviceType = usePlatformStore((state) => state.deviceType);
  const {
    datasetEvalSplit,
    datasetFormat,
    datasetSliceEnd,
    datasetSliceStart,
    datasetSource,
    datasetSplit,
    datasetStreaming,
    evalSteps,
    isAudioModel,
    isDatasetAudio,
    isDatasetImage,
    isEmbeddingModel,
    isVisionModel,
    maxSteps,
    setDatasetFormat,
    setDatasetSliceEnd,
    setDatasetSliceStart,
    setDatasetStreaming,
    trainOnCompletions,
  } = useTrainingConfigStore(
    useShallow((state) => ({
      datasetEvalSplit: state.datasetEvalSplit,
      datasetFormat: state.datasetFormat,
      datasetSliceEnd: state.datasetSliceEnd,
      datasetSliceStart: state.datasetSliceStart,
      datasetSource: state.datasetSource,
      datasetSplit: state.datasetSplit,
      datasetStreaming: state.datasetStreaming,
      evalSteps: state.evalSteps,
      isAudioModel: state.isAudioModel,
      isDatasetAudio: state.isDatasetAudio,
      isDatasetImage: state.isDatasetImage,
      isEmbeddingModel: state.isEmbeddingModel,
      isVisionModel: state.isVisionModel,
      maxSteps: state.maxSteps,
      setDatasetFormat: state.setDatasetFormat,
      setDatasetSliceEnd: state.setDatasetSliceEnd,
      setDatasetSliceStart: state.setDatasetSliceStart,
      setDatasetStreaming: state.setDatasetStreaming,
      trainOnCompletions: state.trainOnCompletions,
    })),
  );
  const streamingBlockers = getDatasetStreamingBlockers({
    datasetEvalSplit,
    datasetSource,
    datasetSplit,
    evalSteps,
    isAppleSilicon: platformDeviceType === "mac",
    isAudioModel,
    isDatasetAudio,
    isDatasetImage,
    isEmbeddingModel,
    isVisionModel,
    maxSteps,
    trainOnCompletions,
  });
  const isStreamingSupported = streamingBlockers.length === 0;

  useEffect(() => {
    if (!datasetStreaming || isStreamingSupported) {
      return;
    }
    setDatasetStreaming(false);
    if (isDatasetImage || isDatasetAudio) {
      toast.info(
        t("studio.dataset.streaming.notifications.disabledForDetectedModality"),
      );
    }
  }, [
    isStreamingSupported,
    datasetStreaming,
    isDatasetAudio,
    isDatasetImage,
    setDatasetStreaming,
    t,
  ]);

  return (
    <DatasetAdvancedSettings
      datasetFormat={datasetFormat}
      datasetSliceEnd={datasetSliceEnd}
      datasetSliceStart={datasetSliceStart}
      datasetStreaming={datasetStreaming}
      isStreamingSupported={isStreamingSupported}
      setDatasetFormat={setDatasetFormat}
      setDatasetSliceEnd={setDatasetSliceEnd}
      setDatasetSliceStart={setDatasetSliceStart}
      setDatasetStreaming={setDatasetStreaming}
      streamingBlockers={streamingBlockers}
    />
  );
}
