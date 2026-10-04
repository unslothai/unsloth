// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AudioApiError,
  type AudioRunResponse,
  fetchAudioBlob,
  runAudio,
} from "../api";
import {
  type AudioOptionSpec,
  type AudioOptionValues,
  audioOptionLabel,
  audioOptionsForRequest,
  missingRequiredAudioOptions,
} from "../audio-options";
import { persistedClipForGeneration } from "../audio-page-policy";
import {
  type AudioSourceSelection,
  selectionExpired,
  sourceRefOf,
} from "../audio-run-request";
import { cloneBlocker, referenceTextField } from "../clone-policy";
import type { AudioSourceInputHandle } from "../components/audio-source-input";
import type { GenerateBlocker } from "../pages/tts-workspace";
import { useAudioCloneStore } from "../stores/audio-clone-store";
import { audioToolPanelsFor } from "../tools/registry";
import {
  audioModelContextFor,
  claimedOptionNames,
  collectToolRequest,
  toolValueKey,
} from "../tools/select";
import type { AudioHostState } from "./audio-host-state";
import type { AudioSourceStatus } from "./audio-source-state";
import { REFERENCE_EXPIRED_MESSAGE } from "./audio-source-state";
import type { AudioGallery } from "./use-audio-gallery";
import { galleryCache } from "./use-audio-gallery";
import type { AudioModelSlot } from "./use-audio-model-slot";
import { useReferenceTranscribe } from "./use-reference-transcribe";

export const CLONE_TEXT_FIELD_ID = "clone-text";
export const CLONE_REFERENCE_TEXT_FIELD_ID = "clone-reference-text";

export async function showRunResult({
  response,
  text,
  workflow,
  refreshGallery,
  selectClip,
  setFallbackClip,
  setSelectedId,
}: {
  response: AudioRunResponse;
  text: string;
  workflow: "speak" | "clone" | "music";
} & Pick<
  AudioGallery,
  "refreshGallery" | "selectClip" | "setFallbackClip" | "setSelectedId"
>): Promise<void> {
  const clip =
    response.clips.find((item) => item.role === "output") ?? response.clips[0];
  const refreshed = await refreshGallery();
  if (clip) {
    const listed = persistedClipForGeneration(clip.id, refreshed);
    if (listed) {
      setFallbackClip(null);
      selectClip(listed.id);
      return;
    }
    // Saved, but this refresh missed it: play the saved file until a later refresh lists it.
    try {
      const blob = await fetchAudioBlob(clip.url);
      setFallbackClip({
        url: URL.createObjectURL(blob),
        prompt: text,
        model: response.model,
        saved: true,
        workflow,
      });
    } catch {}
    selectClip(clip.id, true);
    return;
  }
  if (response.audio) {
    galleryCache.selectedId = null;
    setSelectedId(null);
    setFallbackClip({
      url: `data:audio/${response.audio.format || "wav"};base64,${response.audio.data}`,
      prompt: text,
      model: response.model,
      saved: false,
      workflow,
    });
  }
}

/** Mirrors useSpeechGeneration's flow so Stop, phases and errors behave the same. */
export function useCloneGeneration({
  status,
  busyRef,
  setBusy,
  updateGenerationPhase,
  generateAbort,
  setMode,
  setAdvancedOpen,
  refreshStatus,
  activeRef,
  modeRef,
  refreshGallery,
  selectClip,
  setFallbackClip,
  setSelectedId,
  pendingTranscribeRelease,
  replayQueuedTtsPick,
  audioOptionSpecs,
  audioOptionValues,
  sttRepo,
}: Pick<
  AudioHostState,
  | "status"
  | "busyRef"
  | "setBusy"
  | "updateGenerationPhase"
  | "generateAbort"
  | "setMode"
  | "setAdvancedOpen"
  | "refreshStatus"
  | "activeRef"
  | "modeRef"
> &
  Pick<
    AudioGallery,
    "refreshGallery" | "selectClip" | "setFallbackClip" | "setSelectedId"
  > &
  Pick<AudioModelSlot, "pendingTranscribeRelease" | "replayQueuedTtsPick"> & {
    audioOptionSpecs: AudioOptionSpec[];
    audioOptionValues: AudioOptionValues;
    sttRepo: string | null;
  }) {
  const reference = useAudioCloneStore((state) => state.reference);
  const referenceText = useAudioCloneStore((state) => state.referenceText);
  const text = useAudioCloneStore((state) => state.text);
  const language = useAudioCloneStore((state) => state.language);
  const storedToolValues = useAudioCloneStore((state) => state.toolValues);
  const model = status?.active_model ?? null;
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [referenceStatus, setReferenceStatus] = useState<AudioSourceStatus>({
    phase: "idle",
  });
  const referenceHandle = useRef<AudioSourceInputHandle | null>(null);
  const [expiredReferenceId, setExpiredReferenceId] = useState<string | null>(
    null,
  );

  const referenceTranscriber = useReferenceTranscribe({
    sttRepo,
    onText: (next, source) =>
      useAudioCloneStore.getState().applyTranscript(source, next),
  });
  // Holds the page busy like the Transcribe page does, so Generate and model swaps wait for the STT run.
  const transcribeReference = useCallback(
    async (source: AudioSourceSelection | null) => {
      if (!source || busyRef.current) return;
      busyRef.current = "transcribing";
      setBusy("transcribing");
      try {
        await referenceTranscriber.transcribe(source);
      } finally {
        if (busyRef.current === "transcribing") {
          busyRef.current = null;
          setBusy(null);
        }
      }
    },
    [busyRef, setBusy, referenceTranscriber.transcribe],
  );
  const transcriber = {
    ...referenceTranscriber,
    transcribe: transcribeReference,
  };

  const toolContext = useMemo(
    () =>
      audioModelContextFor(status, {
        musicGeneration: false,
        cudaMusicGeneration: false,
        musicNeedsDescription: false,
      }),
    [status],
  );
  const toolPanels = useMemo(
    () => audioToolPanelsFor("clone", toolContext),
    [toolContext],
  );
  const toolValues = useMemo(() => {
    const values: Record<string, unknown> = {};
    for (const panel of toolPanels) {
      values[panel.id] =
        storedToolValues[toolValueKey(model, "clone", panel.id)];
    }
    return values;
  }, [toolPanels, storedToolValues, model]);
  const handleToolValueChange = useCallback(
    (panelId: string, value: unknown) =>
      useAudioCloneStore
        .getState()
        .setToolValue(toolValueKey(model, "clone", panelId), value),
    [model],
  );
  const toolRequest = useMemo(
    () =>
      collectToolRequest(
        toolPanels,
        toolValues,
        { text, referenceText, hasReference: reference !== null },
        toolContext,
        audioOptionSpecs,
      ),
    [
      toolPanels,
      toolValues,
      text,
      referenceText,
      reference,
      toolContext,
      audioOptionSpecs,
    ],
  );
  const claimedOptions = useMemo(
    () => claimedOptionNames(toolPanels),
    [toolPanels],
  );
  const advancedOptionSpecs = useMemo(
    () => audioOptionSpecs.filter((spec) => !claimedOptions.has(spec.name)),
    [audioOptionSpecs, claimedOptions],
  );
  const transcriptField = referenceTextField(toolContext, toolRequest.patch);

  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (reference?.kind !== "input" || !reference.expiresAt) return;
    const timer = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, [reference]);
  const referenceExpired =
    referenceStatus.phase === "expired" ||
    selectionExpired(reference, now) ||
    (reference !== null && reference.id === expiredReferenceId);

  const inputBlocker = cloneBlocker({
    reference,
    referenceBusy:
      referenceStatus.phase === "uploading" ||
      referenceStatus.phase === "recording",
    referenceExpired,
    referenceError:
      referenceStatus.phase === "error" ? referenceStatus.message : null,
    referenceText,
    referenceTextField: transcriptField,
    text,
    panelError: toolRequest.error,
  });
  const focusField = (id: string) => () => document.getElementById(id)?.focus();
  const addReference = {
    label: "Add reference audio",
    onClick: () => referenceHandle.current?.focus(),
  };
  // Model blockers (none loaded, cannot clone) are the host's and come first.
  const blockerActions: Record<string, GenerateBlocker["actions"]> = {
    reference: [addReference],
    "reference-error": [addReference],
    "reference-expired": [
      {
        label: "Add it again",
        onClick: () => {
          // Like Remove: the expired clip's transcript goes with it.
          useAudioCloneStore.getState().adoptReference(null);
          referenceHandle.current?.browse();
        },
      },
    ],
    "reference-text": [
      {
        label: "Transcribe it",
        onClick: () => void transcriber.transcribe(reference),
      },
      { label: "type it", onClick: focusField(CLONE_REFERENCE_TEXT_FIELD_ID) },
    ],
    text: [{ label: "Write it", onClick: focusField(CLONE_TEXT_FIELD_ID) }],
  };
  const blocker: GenerateBlocker | null = inputBlocker
    ? {
        reason: inputBlocker.reason,
        actions: blockerActions[inputBlocker.kind],
      }
    : null;

  const handleGenerate = useCallback(async () => {
    const state = useAudioCloneStore.getState();
    const draftText = state.text.trim();
    if (!(draftText && state.reference) || inputBlocker) return;
    if (busyRef.current) return;
    busyRef.current = "generating";
    setBusy("generating");
    updateGenerationPhase("preparing");
    const finishEarly = () => {
      updateGenerationPhase(null);
      busyRef.current = null;
      setBusy(null);
    };
    const releaseInFlight = pendingTranscribeRelease.current;
    if (releaseInFlight && !(await releaseInFlight)) {
      finishEarly();
      setMode("transcribe");
      return;
    }
    const missingOptions = missingRequiredAudioOptions(
      advancedOptionSpecs,
      audioOptionValues,
    );
    if (missingOptions.length > 0) {
      finishEarly();
      setAdvancedOpen(true);
      toast.error(
        `Set ${missingOptions.map((spec) => audioOptionLabel(spec.name)).join(", ")} in Advanced before generating.`,
      );
      return;
    }
    const patch = toolRequest.patch;
    const controller = new AbortController();
    generateAbort.current = controller;
    updateGenerationPhase("generating");
    try {
      const response = await runAudio(
        {
          workflow: "clone",
          text: draftText,
          language: state.language || null,
          instructions: patch.instructions ?? null,
          inputs: {
            reference: sourceRefOf(state.reference),
            reference_text:
              transcriptField === "hidden" ? null : state.referenceText,
            emotion: patch.inputs?.emotion ?? null,
          },
          options: {
            ...audioOptionsForRequest(advancedOptionSpecs, audioOptionValues),
            ...patch.options,
          },
          speed: patch.speed ?? null,
        },
        controller.signal,
      );
      updateGenerationPhase("finishing");
      await showRunResult({
        response,
        text: draftText,
        workflow: "clone",
        refreshGallery,
        selectClip,
        setFallbackClip,
        setSelectedId,
      });
    } catch (error) {
      if (!controller.signal.aborted) {
        updateGenerationPhase("finishing");
        const expired =
          error instanceof AudioApiError &&
          error.status === 404 &&
          state.reference.kind === "input" &&
          // With an emotion clip too, the 404 may be that clip's: keep the server's message.
          !patch.inputs?.emotion;
        if (expired) {
          setExpiredReferenceId(state.reference.id);
          referenceHandle.current?.markExpired();
        }
        const message = expired
          ? REFERENCE_EXPIRED_MESSAGE
          : error instanceof Error
            ? error.message
            : "Voice cloning failed.";
        // Kept under Generate so the reason outlives the toast; expiry is already on the card.
        setGenerationError(message);
        if (!expired) toast.error(message);
        await refreshStatus();
      }
    } finally {
      generateAbort.current = null;
      updateGenerationPhase(null);
      busyRef.current = null;
      setBusy(null);
      if (activeRef.current && modeRef.current === "speak")
        replayQueuedTtsPick();
    }
  }, [
    inputBlocker,
    busyRef,
    setBusy,
    updateGenerationPhase,
    pendingTranscribeRelease,
    setMode,
    advancedOptionSpecs,
    audioOptionValues,
    setAdvancedOpen,
    toolRequest,
    generateAbort,
    transcriptField,
    refreshGallery,
    selectClip,
    setFallbackClip,
    setSelectedId,
    refreshStatus,
    activeRef,
    modeRef,
    replayQueuedTtsPick,
  ]);

  useEffect(() => {
    if (reference) setGenerationError(null);
  }, [reference]);

  return {
    reference,
    referenceText,
    text,
    language,
    referenceHandle,
    referenceStatus,
    setReferenceStatus,
    referenceExpired,
    transcriptField,
    transcriber,
    toolContext,
    toolPanels,
    toolValues,
    handleToolValueChange,
    claimedOptions,
    advancedOptionSpecs,
    blocker,
    inputsReady: inputBlocker === null,
    handleGenerate,
    generationError,
    setGenerationError,
  };
}

export type CloneGeneration = ReturnType<typeof useCloneGeneration>;
