// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
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
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { type FC, useMemo, useState } from "react";

interface SttModelSelectorProps {
  /** The transcription model the loop listens with (a dictation key or an audio.cpp folder id). */
  value: string;
  onValueChange: (id: string) => void;
  /** No chat model yet: listening has nothing to reply with. */
  disabled?: boolean;
  /** The listener is loaded, so the row shows as resident. */
  ready?: boolean;
  loading?: boolean;
  className?: string;
}

const transcribeRow = (row: Parameters<typeof audioRowMatchesWorkflow>[0]) =>
  audioRowMatchesWorkflow(row, "transcribe");

/** The picker row for a saved dictation key: the Transcribe tab lists repos, dictation saves keys. */
function sttRepoForKey(key: string): string {
  if (isAudioCppFolderId(key)) return key;
  if (key.startsWith("audiocpp-"))
    return sttRepoIdForSidecarKey(key, "audiocpp");
  if (MTMD_STT_MODELS.has(key as SttModel))
    return sttRepoIdForSidecarKey(key, "mtmd");
  return sttRepoIdForSidecarKey(key, "transformers");
}

/** Voice mode's "Listen with" picker: the Audio tab's Transcribe picker, saving the dictation key. */
export const SttModelSelector: FC<SttModelSelectorProps> = ({
  value,
  onValueChange,
  disabled = false,
  ready = false,
  loading = false,
  className,
}) => {
  const [open, setOpen] = useState(false);
  const models = useMemo(() => audioModelsForTask("stt"), []);
  const repo = sttRepoForKey(value);

  return (
    <div
      className={cn(
        "flex min-w-0 items-center gap-1",
        disabled && "pointer-events-none opacity-50",
      )}
      aria-disabled={disabled || undefined}
      title={disabled ? "Select a chat model first" : undefined}
    >
      {loading && <Spinner className="size-3.5 shrink-0 text-amber-500" />}
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
            toast.info("Voice mode can't listen with that model yet.");
            return;
          }
          onValueChange(key);
        }}
        variant="ghost"
        className={cn("!h-8.5", className)}
        task={["automatic-speech-recognition"]}
        catalog={AUDIO_CATALOG}
        placeholder="Select listening model"
        open={!disabled && open}
        onOpenChange={(next) => setOpen(!disabled && next)}
      />
    </div>
  );
};
