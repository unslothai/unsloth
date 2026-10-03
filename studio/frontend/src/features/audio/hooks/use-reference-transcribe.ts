// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { sttEngineFor } from "@/features/chat/adapters/studio-model-dictation-adapter";
import { useVoiceSettingsStore } from "@/features/settings";
import { useCallback, useEffect, useRef, useState } from "react";
import { transcribeAudioInput } from "../api";
import { type AudioSourceSelection, sourceRefOf } from "../audio-run-request";
import { sttEngineForRepoId, sttSidecarKeyFor } from "../catalog";

/** Fills "What's said in the clip" without adding it to the transcript list. No language hint:
 *  the page's language is the output's, and a cross-lingual reference is not in it. */
export function useReferenceTranscribe({
  sttRepo,
  onText,
  purpose = "reference",
}: {
  sttRepo: string | null;
  onText: (text: string, reference: AudioSourceSelection) => void;
  /** "convert" transcribes up to Convert's cap instead of the 30 s clone reference. */
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
        const voice = useVoiceSettingsStore.getState();
        const model = sttRepo ? sttSidecarKeyFor(sttRepo) : voice.sttModel;
        const result = await transcribeAudioInput(
          sourceRefOf(reference),
          {
            model,
            engine: sttRepo ? sttEngineForRepoId(sttRepo) : sttEngineFor(model),
            device: voice.sttDevice,
            purpose,
          },
          controller.signal,
        );
        if (controller.signal.aborted) return;
        const text = result.text.trim();
        if (text) {
          onTextRef.current(text, reference);
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
    [sttRepo, purpose],
  );

  useEffect(() => () => abort.current?.abort(), []);

  return { transcribe, transcribing, error };
}
