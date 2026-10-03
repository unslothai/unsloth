// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AudioApiError, runAudio } from "../api";
import {
  type AudioOptionSpec,
  type AudioOptionValues,
  audioOptionLabel,
  audioOptionsForRequest,
  missingRequiredAudioOptions,
} from "../audio-options";
import {
  type AudioSourceSelection,
  selectionExpired,
  sourceRefOf,
} from "../audio-run-request";
import type { AudioSourceInputHandle } from "../components/audio-source-input";
import { buildEditRun, editAdapterFor, editPhaseLabel } from "../edit-adapters";
import { type EditBlockerActionId, editBlocker } from "../edit-policy";
import type { GenerateBlocker } from "../pages/tts-workspace";
import { useAudioCloneStore } from "../stores/audio-clone-store";
import { useAudioEditStore } from "../stores/audio-edit-store";
import { audioToolPanelsFor } from "../tools/registry";
import {
  audioModelContextFor,
  claimedOptionNames,
  collectToolRequest,
  toolValueKey,
} from "../tools/select";
import type { EditCoreInputs } from "../tools/types";
import type { AudioHostState } from "./audio-host-state";
import type { AudioSourceStatus } from "./audio-source-state";
import { REFERENCE_EXPIRED_MESSAGE } from "./audio-source-state";
import type { AudioGallery } from "./use-audio-gallery";
import type { AudioModelSlot } from "./use-audio-model-slot";
import { showRunResult } from "./use-clone-generation";
import { useReferenceTranscribe } from "./use-reference-transcribe";

export const EDIT_TRANSCRIPT_FIELD_ID = "edit-transcript";
export const EDIT_CHANGES_FIELD_ID = "edit-changes";

/** Picking a recording fills ① with what it already says (a history clip's text); a new one
 *  without text clears the old transcript so it gets transcribed again. */
export function adoptEditSource(next: AudioSourceSelection | null) {
  const store = useAudioEditStore.getState();
  if (next?.id === store.source?.id && next?.kind === store.source?.kind) {
    store.setSource(next);
    return;
  }
  store.setSource(next);
  if (!next) return;
  // A new recording starts its changes over from its own transcript.
  useAudioEditStore.setState({ editedTouched: false });
  store.setTranscript(
    next.transcript?.trim() ?? "",
    next.transcript ? next.id : null,
  );
}

/** Edit: its draft (kept in the edit store), the model's adapter and tools, what holds Generate
 *  back, and the run. Mirrors useCloneGeneration so Stop, phases and errors behave the same. */
export function useEditGeneration({
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
  onOpenTranscribe,
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
    /** Transcribe's selected or last speech-to-text repo, for ①. */
    sttRepo: string | null;
    /** Takes the user to Transcribe to get a speech-to-text model. */
    onOpenTranscribe: () => void;
  }) {
  const source = useAudioEditStore((state) => state.source);
  const transcript = useAudioEditStore((state) => state.transcript);
  const transcriptFor = useAudioEditStore((state) => state.transcriptFor);
  const edited = useAudioEditStore((state) => state.edited);
  const editMode = useAudioEditStore((state) => state.mode);
  const delivery = useAudioEditStore((state) => state.delivery);
  const storedToolValues = useAudioCloneStore((state) => state.toolValues);
  const model = status?.active_model ?? null;
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [phaseLabel, setPhaseLabel] = useState<string | null>(null);
  const [sourceStatus, setSourceStatus] = useState<AudioSourceStatus>({
    phase: "idle",
  });
  const sourceHandle = useRef<AudioSourceInputHandle | null>(null);
  const changesRef = useRef<HTMLTextAreaElement | null>(null);
  const [expiredSourceId, setExpiredSourceId] = useState<string | null>(null);

  const transcriber = useReferenceTranscribe({
    sttRepo,
    language: "",
    onText: (next) => {
      const current = useAudioEditStore.getState().source;
      useAudioEditStore.getState().setTranscript(next, current?.id ?? null);
    },
  });

  const toolContext = useMemo(
    () =>
      audioModelContextFor(status, {
        musicGeneration: false,
        cudaMusicGeneration: false,
        musicNeedsDescription: false,
      }),
    [status],
  );
  const adapter = useMemo(() => editAdapterFor(toolContext), [toolContext]);
  // Delivery shows only for a model that can change it; Words otherwise.
  const mode =
    editMode === "delivery" && adapter?.delivery ? "delivery" : "words";
  const toolPanels = useMemo(
    () => audioToolPanelsFor("edit", toolContext),
    [toolContext],
  );
  const toolValues = useMemo(() => {
    const values: Record<string, unknown> = {};
    for (const panel of toolPanels) {
      values[panel.id] =
        storedToolValues[toolValueKey(model, "edit", panel.id)];
    }
    return values;
  }, [toolPanels, storedToolValues, model]);
  const handleToolValueChange = useCallback(
    (panelId: string, value: unknown) =>
      useAudioCloneStore
        .getState()
        .setToolValue(toolValueKey(model, "edit", panelId), value),
    [model],
  );
  const claimedOptions = useMemo(
    () => claimedOptionNames(toolPanels),
    [toolPanels],
  );
  const advancedOptionSpecs = useMemo(
    () => audioOptionSpecs.filter((spec) => !claimedOptions.has(spec.name)),
    [audioOptionSpecs, claimedOptions],
  );
  const advancedRequest = useMemo(
    () => audioOptionsForRequest(advancedOptionSpecs, audioOptionValues),
    [advancedOptionSpecs, audioOptionValues],
  );
  const core: EditCoreInputs = useMemo(
    () => ({
      transcript,
      edited,
      mode,
      speed: delivery.speed,
      pitchSteps: delivery.pitchSteps,
      advanced: advancedRequest,
    }),
    [transcript, edited, mode, delivery, advancedRequest],
  );
  const toolRequest = useMemo(
    () =>
      collectToolRequest(
        toolPanels,
        toolValues,
        { text: edited, edit: core },
        toolContext,
        audioOptionSpecs,
      ),
    [toolPanels, toolValues, edited, core, toolContext, audioOptionSpecs],
  );

  // An upload can pass its keep-until time while the page sits open.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (source?.kind !== "input" || !source.expiresAt) return;
    const timer = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, [source]);
  const sourceExpired =
    sourceStatus.phase === "expired" ||
    selectionExpired(source, now) ||
    (source !== null && source.id === expiredSourceId);
  const sourceBusy =
    sourceStatus.phase === "uploading" || sourceStatus.phase === "recording";

  // ① fills itself once per recording: speech-to-text runs as soon as the recording is usable.
  const autoTranscribed = useRef<string | null>(null);
  const transcribe = transcriber.transcribe;
  useEffect(() => {
    if (!source || sourceBusy || sourceExpired) return;
    if (transcriptFor === source.id || transcript.trim()) return;
    if (autoTranscribed.current === source.id) return;
    autoTranscribed.current = source.id;
    void transcribe(source);
  }, [
    source,
    sourceBusy,
    sourceExpired,
    transcriptFor,
    transcript,
    transcribe,
  ]);

  const focusField = useCallback((id: string) => {
    document.getElementById(id)?.focus();
  }, []);

  const inputBlocker = editBlocker({
    source,
    sourceBusy,
    sourceExpired,
    sourceError: sourceStatus.phase === "error" ? sourceStatus.message : null,
    sourceDurationS: source?.durationS ?? null,
    transcribing: transcriber.transcribing,
    // The page falls back to the Settings > Voice model, so there is always one to try.
    sttAvailable: true,
    transcript,
    edited,
    mode,
    delivery,
    panelError: toolRequest.error,
  });
  const runAction = useCallback(
    (id: EditBlockerActionId) => {
      switch (id) {
        case "add-recording":
          if (sourceExpired) useAudioEditStore.getState().setSource(null);
          sourceHandle.current?.browse();
          return;
        case "choose-recording":
          useAudioEditStore.getState().setSource(null);
          sourceHandle.current?.browse();
          return;
        case "transcribe":
          void transcribe(useAudioEditStore.getState().source);
          return;
        case "type-transcript":
          focusField(EDIT_TRANSCRIPT_FIELD_ID);
          return;
        case "open-transcribe":
          onOpenTranscribe();
          return;
        case "focus-changes":
          changesRef.current?.focus();
          return;
      }
    },
    [sourceExpired, transcribe, focusField, onOpenTranscribe],
  );
  const blocker: GenerateBlocker | null = inputBlocker
    ? {
        reason: inputBlocker.reason,
        actions: inputBlocker.actions.length
          ? inputBlocker.actions.map((action) => ({
              label: action.label,
              onClick: () => runAction(action.id),
            }))
          : undefined,
      }
    : null;

  const handleGenerate = useCallback(async () => {
    const state = useAudioEditStore.getState();
    if (!(state.source && adapter) || inputBlocker) return;
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
    const run = buildEditRun(adapter, {
      source: sourceRefOf(state.source),
      transcript: state.transcript,
      edited: state.edited,
      mode,
      delivery: state.delivery,
      advanced: { ...advancedRequest, ...toolRequest.patch.options },
      sourceName: state.source.name,
    });
    const controller = new AbortController();
    generateAbort.current = controller;
    setPhaseLabel(editPhaseLabel(adapter, run));
    updateGenerationPhase("generating");
    try {
      const response = await runAudio(run, controller.signal);
      updateGenerationPhase("finishing");
      await showRunResult({
        response,
        text: run.text,
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
          state.source.kind === "input";
        if (expired) {
          setExpiredSourceId(state.source.id);
          sourceHandle.current?.markExpired();
        }
        const message = expired
          ? REFERENCE_EXPIRED_MESSAGE
          : error instanceof Error
            ? error.message
            : "Editing the recording failed.";
        setGenerationError(message);
        if (!expired) toast.error(message);
        await refreshStatus();
      }
    } finally {
      generateAbort.current = null;
      setPhaseLabel(null);
      updateGenerationPhase(null);
      busyRef.current = null;
      setBusy(null);
      if (activeRef.current && modeRef.current === "speak")
        replayQueuedTtsPick();
    }
  }, [
    adapter,
    inputBlocker,
    busyRef,
    setBusy,
    updateGenerationPhase,
    pendingTranscribeRelease,
    setMode,
    advancedOptionSpecs,
    audioOptionValues,
    setAdvancedOpen,
    mode,
    advancedRequest,
    toolRequest,
    generateAbort,
    refreshGallery,
    selectClip,
    setFallbackClip,
    setSelectedId,
    refreshStatus,
    activeRef,
    modeRef,
    replayQueuedTtsPick,
  ]);

  // A new recording clears a failure that was about the old one.
  useEffect(() => {
    if (source) setGenerationError(null);
  }, [source]);

  return {
    source,
    transcript,
    edited,
    mode,
    delivery,
    adapter,
    sourceHandle,
    changesRef,
    sourceStatus,
    setSourceStatus,
    sourceExpired,
    transcriber,
    toolContext,
    toolPanels,
    toolValues,
    handleToolValueChange,
    claimedOptions,
    advancedOptionSpecs,
    core,
    blocker,
    /** Whether the page inputs are complete; the host still checks the model. */
    inputsReady: inputBlocker === null,
    handleGenerate,
    generationError,
    setGenerationError,
    /** "Applying 3 changes, one pass each" while a chained FireRedAudio run is going. */
    phaseLabel,
  };
}

export type EditGeneration = ReturnType<typeof useEditGeneration>;
