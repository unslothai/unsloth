// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isAudioCppFolderId } from "@/features/audio/audio-cpp-catalog";
import { audioModelsForTask } from "@/features/audio/catalog";
import { audioRowMatchesWorkflow } from "@/features/audio/picker-filter";
import { ModelSelector } from "@/features/model-picker";
import { AUDIO_CATALOG } from "@/features/model-picker/components/model-selector/model-catalog";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { type FC, useMemo } from "react";

interface VadModelSelectorProps {
  value: string;
  onValueChange: (id: string) => void;
  loaded?: boolean;
  /** Called once a pick is made or the picker closes. */
  onClose?: () => void;
}

const detectRow = (row: Parameters<typeof audioRowMatchesWorkflow>[0]) =>
  audioRowMatchesWorkflow(row, "detect");

/** Voice mode's "Detect speech with" picker, inline in the voice panel: the audio.cpp voice activity detectors. */
export const VadModelSelector: FC<VadModelSelectorProps> = ({
  value,
  onValueChange,
  loaded = false,
  onClose,
}) => {
  const t = useT();
  const models = useMemo(() => audioModelsForTask("vad"), []);
  return (
    <ModelSelector
      models={models}
      rowFilter={detectRow}
      value={value}
      loaded={loaded}
      loadedModelIdOverride={loaded ? value : undefined}
      onValueChange={(id) => {
        if (!isAudioCppFolderId(id)) {
          toast.info(t("voicePanel.ggufDetectorsOnly"));
          return;
        }
        onValueChange(id);
      }}
      variant="ghost"
      className="!h-8.5"
      task={["voice-activity-detection"]}
      catalog={AUDIO_CATALOG}
      embedded={true}
      open={true}
      onOpenChange={(next) => {
        if (!next) onClose?.();
      }}
    />
  );
};
