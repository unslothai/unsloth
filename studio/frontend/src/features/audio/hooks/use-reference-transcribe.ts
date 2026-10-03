// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { sttEngineFor } from "@/features/chat/adapters/studio-model-dictation-adapter";
import { useVoiceSettingsStore } from "@/features/settings";
import { useCallback, useEffect, useRef, useState } from "react";
import { transcribeAudioInput } from "../api";
import { type AudioSourceSelection, sourceRefOf } from "../audio-run-request";
import { sttEngineForRepoId, sttSidecarKeyFor } from "../catalog";

/** The speech-to-text model the page uses: Transcribe's pick, else the dictation model from
 *  Settings > Voice. */
export function referenceSttModel(sttRepo: string | null): {
  model: string;
  engine: string;
} {
  if (sttRepo) {
    return {
      model: sttSidecarKeyFor(sttRepo),
      engine: sttEngineForRepoId(sttRepo),
    };
  }
  const model = useVoiceSettingsStore.getState().sttModel;
  return { model, engine: sttEngineFor(model) };
}

/** Fills "What's said in the clip" from the reference, without adding it to the transcript list. */
export function useReferenceTranscribe({
  sttRepo,
  language,
  onText,
  purpose = "reference",
}: {
  /** Transcribe's selected or last speech-to-text repo, if any. */
  sttRepo: string | null;
  /** The clone language, as a hint; empty lets the model detect it. */
  language: string;
  onText: (text: string) => void;
  /** "convert" transcribes as much as Convert converts, not the clone reference's 30 s. */
  purpose?: "reference" | "convert";
}) {
  const [transcribing, setTranscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const onTextRef = useRef(onText);
  onTextRef.current = onText;

  const transcribe = useCallback(
    async (reference: AudioSourceSelection | null) => {
      if (!reference || abort.current) return;
      const controller = new AbortController();
      abort.current = controller;
      setTranscribing(true);
      setError(null);
      try {
        const target = referenceSttModel(sttRepo);
        const result = await transcribeAudioInput(
          sourceRefOf(reference),
          {
            ...target,
            device: useVoiceSettingsStore.getState().sttDevice,
            ...(language ? { language } : {}),
            purpose,
          },
          controller.signal,
        );
        if (controller.signal.aborted) return;
        const text = result.text.trim();
        if (text) {
          onTextRef.current(text);
        } else {
          setError(
            "No speech was heard in the clip. Type what's said instead.",
          );
        }
      } catch (reason) {
        if (controller.signal.aborted) return;
        setError(
          reason instanceof Error && reason.message
            ? reason.message
            : "Could not transcribe the clip.",
        );
      } finally {
        if (abort.current === controller) abort.current = null;
        setTranscribing(false);
      }
    },
    [sttRepo, language, purpose],
  );

  const cancel = useCallback(() => {
    abort.current?.abort();
    abort.current = null;
    setTranscribing(false);
  }, []);

  useEffect(() => () => abort.current?.abort(), []);

  return {
    transcribe,
    transcribing,
    error,
    clearError: () => setError(null),
    cancel,
  };
}

export type ReferenceTranscribe = ReturnType<typeof useReferenceTranscribe>;
