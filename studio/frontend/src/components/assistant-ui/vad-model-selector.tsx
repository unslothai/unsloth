// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isAudioCppFolderId } from "@/features/audio/audio-cpp-catalog";
import { audioModelsForTask } from "@/features/audio/catalog";
import { audioRowMatchesWorkflow } from "@/features/audio/picker-filter";
import { useVadDetectors } from "@/features/chat/voice/use-vad-detectors";
import { ModelSelector } from "@/features/model-picker";
import { AUDIO_CATALOG } from "@/features/model-picker/components/model-selector/model-catalog";
// eslint-disable-next-line no-restricted-imports
import { ListLabel, ModelRow } from "@/features/model-picker/components/model-selector/pickers";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { type FC, useMemo } from "react";

interface VadModelSelectorProps {
  value: string;
  onValueChange: (id: string) => void;
  loaded?: boolean;
  onClose?: () => void;
}

const detectRow = (row: Parameters<typeof audioRowMatchesWorkflow>[0]) =>
  audioRowMatchesWorkflow(row, "detect");

const NO_CAPABILITIES = {
  vision: false,
  reasoning: false,
  audio: false,
  imageGen: false,
  videoGen: false,
};

export const VadModelSelector: FC<VadModelSelectorProps> = ({
  value,
  onValueChange,
  loaded = false,
  onClose,
}) => {
  const t = useT();
  const models = useMemo(() => audioModelsForTask("vad"), []);
  const builtin = useVadDetectors().filter((d) => d.available);
  return (
    <div className="flex flex-col">
      {builtin.length > 0 && (
        <div className="w-[min(var(--picker-panel-w),calc(100vw-1rem))] max-w-[calc(100vw-1rem)] pt-2 pl-4 pr-2">
          <ListLabel>{t("voicePanel.builtinDetectors")}</ListLabel>
          {builtin.map((d) => (
            <ModelRow
              key={d.id}
              label={d.label}
              selected={value === d.id}
              loaded={loaded && value === d.id}
              onClick={() => onValueChange(d.id)}
              capabilities={NO_CAPABILITIES}
              hideOwner={true}
              tags={["CPU"]}
            />
          ))}
        </div>
      )}
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
    </div>
  );
};
