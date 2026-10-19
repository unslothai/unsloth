// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isAudioCppFolderId } from "@/features/audio/audio-cpp-catalog";
import { audioModelsForTask } from "@/features/audio/catalog";
import { audioRowMatchesWorkflow } from "@/features/audio/picker-filter";
import {
  sttRepoIdForSidecarKey,
  sttSidecarKeyFor,
} from "@/features/audio/stt-artifacts";
import { ModelSelector } from "@/features/model-picker";
import { AUDIO_CATALOG } from "@/features/model-picker/components/model-selector/model-catalog";
import {
  MTMD_STT_MODELS,
  STT_PICKER_MODELS,
  type SttModel,
} from "@/features/settings/stores/stt-model-catalog";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { type FC, useMemo } from "react";

interface SttModelSelectorProps {
  /** The transcription model the loop listens with (a dictation key or an audio.cpp folder id). */
  value: string;
  onValueChange: (id: string) => void;
  /** The listener is loaded, so the row shows as resident. */
  ready?: boolean;
  /** Called once a pick is made or the picker closes. */
  onClose?: () => void;
}

const transcribeRow = (row: Parameters<typeof audioRowMatchesWorkflow>[0]) =>
  audioRowMatchesWorkflow(row, "transcribe");

/** The picker row for a saved dictation key: the Transcribe tab lists repos, dictation saves keys. */
export function sttRepoForKey(key: string): string {
  if (isAudioCppFolderId(key)) return key;
  if (key.startsWith("audiocpp-"))
    return sttRepoIdForSidecarKey(key, "audiocpp");
  if (MTMD_STT_MODELS.has(key as SttModel))
    return sttRepoIdForSidecarKey(key, "mtmd");
  return sttRepoIdForSidecarKey(key, "transformers");
}

/** Voice mode's "Listen with" picker, inline in the voice panel: the Audio tab's Transcribe picker, saving the dictation key. */
export const SttModelSelector: FC<SttModelSelectorProps> = ({
  value,
  onValueChange,
  ready = false,
  onClose,
}) => {
  const t = useT();
  const models = useMemo(() => audioModelsForTask("stt"), []);
  const repo = sttRepoForKey(value);
  return (
    <ModelSelector
      models={models}
      rowFilter={transcribeRow}
      value={repo}
      loaded={ready}
      loadedModelIdOverride={ready ? repo : undefined}
      onValueChange={(picked) => {
        const key = sttSidecarKeyFor(picked);
        if (
          !isAudioCppFolderId(key) &&
          !STT_PICKER_MODELS.includes(key as SttModel)
        ) {
          toast.info(t("voicePanel.cantListenWith"));
          return;
        }
        onValueChange(key);
      }}
      variant="ghost"
      className="!h-8.5"
      task={["automatic-speech-recognition"]}
      catalog={AUDIO_CATALOG}
      embedded={true}
      open={true}
      onOpenChange={(next) => {
        if (!next) onClose?.();
      }}
    />
  );
};
