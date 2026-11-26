// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useHfTokenStore } from "@/features/hub";
import { fetchDefaultChatTemplate } from "@/features/model-picker/api/templates";
import { ChatTemplateEditorDialog } from "@/features/model-picker/components/chat-template-editor-dialog";
import { useTrainingConfigStore } from "@/features/training";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useEffect, useState } from "react";
import { useShallow } from "zustand/react/shallow";
import { DatasetAdvancedSettings } from "./dataset-advanced-settings";
import { getDatasetStreamingBlockers } from "./dataset-panel-helpers";

export function DatasetAdvancedSettingsSection() {
  const t = useT();
  const {
    chatTemplate,
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
    isDecisionModel,
    isVisionModel,
    maxSteps,
    selectedModel,
    setChatTemplate,
    setDatasetFormat,
    setDatasetSliceEnd,
    setDatasetSliceStart,
    setDatasetStreaming,
    trainOnCompletions,
  } = useTrainingConfigStore(
    useShallow((state) => ({
      chatTemplate: state.chatTemplate,
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
      isDecisionModel: state.modelType === "decision",
      isVisionModel: state.isVisionModel,
      maxSteps: state.maxSteps,
      selectedModel: state.selectedModel,
      setChatTemplate: state.setChatTemplate,
      setDatasetFormat: state.setDatasetFormat,
      setDatasetSliceEnd: state.setDatasetSliceEnd,
      setDatasetSliceStart: state.setDatasetSliceStart,
      setDatasetStreaming: state.setDatasetStreaming,
      trainOnCompletions: state.trainOnCompletions,
    })),
  );
  const hfToken = useHfTokenStore((state) => state.token);
  const [templateEditorOpen, setTemplateEditorOpen] = useState(false);
  const [defaultTemplate, setDefaultTemplate] = useState<string | null>(null);
  const [defaultTemplateLoading, setDefaultTemplateLoading] = useState(false);

  useEffect(() => {
    if (!templateEditorOpen || !selectedModel) {
      return;
    }
    const controller = new AbortController();
    setDefaultTemplate(null);
    setDefaultTemplateLoading(true);
    fetchDefaultChatTemplate(selectedModel, null, hfToken, controller.signal)
      .then((template) => {
        if (!controller.signal.aborted) {
          setDefaultTemplate(template);
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setDefaultTemplate(null);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setDefaultTemplateLoading(false);
        }
      });
    return () => controller.abort();
  }, [templateEditorOpen, selectedModel, hfToken]);

  const streamingBlockers = getDatasetStreamingBlockers({
    datasetEvalSplit,
    datasetSource,
    datasetSplit,
    evalSteps,
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
    <>
      <DatasetAdvancedSettings
        chatTemplate={chatTemplate}
        datasetFormat={datasetFormat}
        datasetSliceEnd={datasetSliceEnd}
        datasetSliceStart={datasetSliceStart}
        datasetStreaming={datasetStreaming}
        isDecisionModel={isDecisionModel}
        isEmbeddingModel={isEmbeddingModel}
        isStreamingSupported={isStreamingSupported}
        onEditChatTemplate={() => setTemplateEditorOpen(true)}
        setDatasetFormat={setDatasetFormat}
        setDatasetSliceEnd={setDatasetSliceEnd}
        setDatasetSliceStart={setDatasetSliceStart}
        setDatasetStreaming={setDatasetStreaming}
        streamingBlockers={streamingBlockers}
      />
      <ChatTemplateEditorDialog
        open={templateEditorOpen}
        onOpenChange={setTemplateEditorOpen}
        value={chatTemplate.trim() ? chatTemplate : null}
        defaultTemplate={defaultTemplate}
        defaultLoading={defaultTemplateLoading}
        description={t("studio.dataset.chatTemplateEditorDescription")}
        onSave={(override) => setChatTemplate(override ?? "")}
      />
    </>
  );
}
