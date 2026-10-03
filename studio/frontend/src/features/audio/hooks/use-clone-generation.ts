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
import { selectionExpired, sourceRefOf } from "../audio-run-request";
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

/** Selects what a /audio/run produced, the way a /audio/generate result is selected: the saved
 *  clip once the gallery lists it, else its bytes, so an expensive run is never dropped. */
export async function showRunResult({
  response,
  text,
  refreshGallery,
  selectClip,
  setFallbackClip,
  setSelectedId,
}: {
  response: AudioRunResponse;
  text: string;
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
      });
    } catch {
      // The id is still selected below; the next refresh shows it.
    }
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
    });
  }
}

/** Clone: its draft (kept in the clone store), the model's tools, what holds Generate back, and
 *  the run. Mirrors useSpeechGeneration's flow so Stop, phases and errors behave the same. */
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
    /** The loaded model's Advanced schema and values, shared with Speak (useSpeechGeneration). */
    audioOptionSpecs: AudioOptionSpec[];
    audioOptionValues: AudioOptionValues;
    /** Transcribe's selected or last speech-to-text repo, for the Transcribe button. */
    sttRepo: string | null;
  }) {
  const reference = useAudioCloneStore((state) => state.reference);
  const referenceText = useAudioCloneStore((state) => state.referenceText);
  const text = useAudioCloneStore((state) => state.text);
  const language = useAudioCloneStore((state) => state.language);
  const storedToolValues = useAudioCloneStore((state) => state.toolValues);
  const model = status?.active_model ?? null;
  const [generationError, setGenerationError] = useState<string | null>(null);
  // What the reference card is doing, reported by the card itself.
  const [referenceStatus, setReferenceStatus] = useState<AudioSourceStatus>({
    phase: "idle",
  });
  const referenceHandle = useRef<AudioSourceInputHandle | null>(null);
  // An upload a run found gone, by id, so picking another one clears it.
  const [expiredReferenceId, setExpiredReferenceId] = useState<string | null>(
    null,
  );

  const transcriber = useReferenceTranscribe({
    sttRepo,
    language,
    onText: (next) => useAudioCloneStore.getState().setReferenceText(next),
  });
  // A transcription still running for the previous reference must not fill in the new one's text.
  const referenceKey = reference ? `${reference.kind}:${reference.id}` : null;
  const cancelTranscribe = transcriber.cancel;
  useEffect(() => cancelTranscribe(), [referenceKey, cancelTranscribe]);

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

  // An upload can pass its keep-until time while the page sits open.
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

  const focusReference = useCallback(() => {
    referenceHandle.current?.focus();
  }, []);
  const focusField = useCallback((id: string) => {
    document.getElementById(id)?.focus();
  }, []);

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
  /** What holds Generate back on this page, with the fix as an action. Model blockers (none
   *  loaded, cannot clone) are the host's and come first. */
  const blocker: GenerateBlocker | null = inputBlocker
    ? {
        reason: inputBlocker.reason,
        actions:
          inputBlocker.kind === "reference" ||
          inputBlocker.kind === "reference-error"
            ? [{ label: "Add reference audio", onClick: focusReference }]
            : inputBlocker.kind === "reference-expired"
              ? [
                  {
                    label: "Add it again",
                    onClick: () => {
                      useAudioCloneStore.getState().setReference(null);
                      referenceHandle.current?.browse();
                    },
                  },
                ]
              : inputBlocker.kind === "reference-text"
                ? [
                    {
                      label: "Transcribe it",
                      onClick: () => void transcriber.transcribe(reference),
                    },
                    {
                      label: "type it",
                      onClick: () => focusField(CLONE_REFERENCE_TEXT_FIELD_ID),
                    },
                  ]
                : inputBlocker.kind === "text"
                  ? [
                      {
                        label: "Write it",
                        onClick: () => focusField(CLONE_TEXT_FIELD_ID),
                      },
                    ]
                  : undefined,
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
        refreshGallery,
        selectClip,
        setFallbackClip,
        setSelectedId,
      });
      // The run can restart audio.cpp under another task (Chatterbox: clon after vc), which
      // Convert reads to say whether its next run reloads.
      await refreshStatus();
    } catch (error) {
      if (!controller.signal.aborted) {
        updateGenerationPhase("finishing");
        const expired =
          error instanceof AudioApiError &&
          error.status === 404 &&
          state.reference.kind === "input";
        if (expired) {
          setExpiredReferenceId(state.reference.id);
          referenceHandle.current?.markExpired();
        }
        const message = expired
          ? REFERENCE_EXPIRED_MESSAGE
          : error instanceof Error
            ? error.message
            : "Voice cloning failed.";
        // Kept under Generate too, so the reason outlives the toast. An expired reference is
        // already marked on its card and under Generate.
        setGenerationError(message);
        if (!expired) toast.error(message);
      }
      // Also after Stop: the run may already have restarted audio.cpp under another task.
      await refreshStatus();
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

  // A new reference clears a failure that was about the old one.
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
    /** Whether the page inputs are complete; the host still checks the model. */
    inputsReady: inputBlocker === null,
    handleGenerate,
    generationError,
    setGenerationError,
  };
}

export type CloneGeneration = ReturnType<typeof useCloneGeneration>;
