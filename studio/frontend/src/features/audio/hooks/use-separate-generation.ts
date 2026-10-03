// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AudioApiError, runAudio } from "../api";
import { selectionExpired, sourceRefOf } from "../audio-run-request";
import type { AudioSourceInputHandle } from "../components/audio-source-input";
import type { GenerateBlocker } from "../pages/tts-workspace";
import {
  type OverlapValue,
  type SeparateBlockerKind,
  estimateSeparateSeconds,
  overlapReloads,
  separateBlocker,
  separateCpuWarning,
} from "../separate-policy";
import { useAudioCloneStore } from "../stores/audio-clone-store";
import { useAudioSeparateStore } from "../stores/audio-separate-store";
import { audioToolPanelsFor } from "../tools/registry";
import {
  audioModelContextFor,
  collectToolRequest,
  panelValue,
  toolValueKey,
} from "../tools/select";
import type { AudioHostState } from "./audio-host-state";
import type { AudioSourceStatus } from "./audio-source-state";
import type { AudioGallery } from "./use-audio-gallery";
import type { AudioModelSlot } from "./use-audio-model-slot";
import { showRunResult } from "./use-clone-generation";

export const SEPARATE_TRACK_EXPIRED_MESSAGE =
  "This track expired. Add it again.";

/** Mirrors useCloneGeneration so Stop, phases and errors behave the same. */
export function useSeparateGeneration({
  status,
  busyRef,
  setBusy,
  updateGenerationPhase,
  generateAbort,
  setMode,
  refreshStatus,
  activeRef,
  modeRef,
  audioDevice,
  refreshGallery,
  selectClip,
  setFallbackClip,
  setSelectedId,
  pendingTranscribeRelease,
  replayQueuedTtsPick,
}: Pick<
  AudioHostState,
  | "status"
  | "busyRef"
  | "setBusy"
  | "updateGenerationPhase"
  | "generateAbort"
  | "setMode"
  | "refreshStatus"
  | "activeRef"
  | "modeRef"
  | "audioDevice"
> &
  Pick<
    AudioGallery,
    "refreshGallery" | "selectClip" | "setFallbackClip" | "setSelectedId"
  > &
  Pick<AudioModelSlot, "pendingTranscribeRelease" | "replayQueuedTtsPick">) {
  const source = useAudioSeparateStore((state) => state.source);
  const lastOverlapByModel = useAudioSeparateStore(
    (state) => state.lastOverlapByModel,
  );
  const storedToolValues = useAudioCloneStore((state) => state.toolValues);
  const model = status?.active_model ?? null;
  const family = status?.audio_family ?? null;
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [sourceStatus, setSourceStatus] = useState<AudioSourceStatus>({
    phase: "idle",
  });
  const sourceHandle = useRef<AudioSourceInputHandle | null>(null);
  const [expiredSourceId, setExpiredSourceId] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<{
    groupId: string | null;
    stems: number;
  } | null>(null);
  const [reloading, setReloading] = useState(false);
  const [pendingRun, setPendingRun] = useState<{
    title: string;
    model: string | null;
  } | null>(null);

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
    () => audioToolPanelsFor("separate", toolContext),
    [toolContext],
  );
  const toolValues = useMemo(() => {
    const values: Record<string, unknown> = {};
    for (const panel of toolPanels) {
      values[panel.id] =
        storedToolValues[toolValueKey(model, "separate", panel.id)];
    }
    return values;
  }, [toolPanels, storedToolValues, model]);
  const handleToolValueChange = useCallback(
    (panelId: string, value: unknown) =>
      useAudioCloneStore
        .getState()
        .setToolValue(toolValueKey(model, "separate", panelId), value),
    [model],
  );
  const toolRequest = useMemo(
    () => collectToolRequest(toolPanels, toolValues, { text: "" }, toolContext),
    [toolPanels, toolValues, toolContext],
  );
  const overlapPanel = toolPanels.find(
    (panel) => panel.id === "roformer-overlap",
  );
  const overlap = overlapPanel
    ? panelValue<OverlapValue>(overlapPanel, toolValues, []).overlap
    : true;
  const willReload =
    overlapPanel !== undefined &&
    model !== null &&
    overlapReloads(lastOverlapByModel[model], overlap);

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

  const inputBlocker = separateBlocker({
    source,
    sourceBusy:
      sourceStatus.phase === "uploading" || sourceStatus.phase === "recording",
    sourceExpired,
    sourceError: sourceStatus.phase === "error" ? sourceStatus.message : null,
  });
  const addTrack = {
    label: "Add a track",
    onClick: () => sourceHandle.current?.focus(),
  };
  const blockerActions: Partial<
    Record<SeparateBlockerKind, GenerateBlocker["actions"]>
  > = {
    source: [addTrack],
    "source-error": [addTrack],
    "source-expired": [
      {
        label: "Add it again",
        onClick: () => {
          useAudioSeparateStore.getState().setSource(null);
          sourceHandle.current?.browse();
        },
      },
    ],
    "too-long": [
      {
        label: "Pick a shorter track",
        onClick: () => sourceHandle.current?.browse(),
      },
    ],
  };
  const blocker: GenerateBlocker | null = inputBlocker
    ? {
        reason: inputBlocker.reason,
        actions: blockerActions[inputBlocker.kind],
      }
    : null;

  const estimateSeconds = estimateSeparateSeconds(
    family,
    source?.durationS,
    overlap,
  );
  const cpuWarning = separateCpuWarning(family, audioDevice);

  const handleGenerate = useCallback(async () => {
    const state = useAudioSeparateStore.getState();
    if (!state.source || inputBlocker) return;
    if (busyRef.current) return;
    busyRef.current = "generating";
    setBusy("generating");
    setLastResult(null);
    updateGenerationPhase("preparing");
    const releaseInFlight = pendingTranscribeRelease.current;
    if (releaseInFlight && !(await releaseInFlight)) {
      updateGenerationPhase(null);
      busyRef.current = null;
      setBusy(null);
      setMode("transcribe");
      return;
    }
    const runModel = model;
    const controller = new AbortController();
    generateAbort.current = controller;
    setReloading(willReload);
    setPendingRun({ title: state.source.name, model: runModel });
    updateGenerationPhase("generating");
    try {
      const response = await runAudio(
        {
          workflow: "separate",
          inputs: { source: sourceRefOf(state.source) },
          options: toolRequest.patch.options,
        },
        controller.signal,
      );
      if (runModel && overlapPanel) {
        useAudioSeparateStore.getState().setLastOverlap(runModel, overlap);
      }
      updateGenerationPhase("finishing");
      await showRunResult({
        response,
        text: state.source.name,
        refreshGallery,
        selectClip,
        setFallbackClip,
        setSelectedId,
        workflow: "separate",
      });
      setLastResult({
        groupId: response.group_id,
        stems: response.clips.length,
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
          ? SEPARATE_TRACK_EXPIRED_MESSAGE
          : error instanceof Error
            ? error.message
            : "Separation failed.";
        setGenerationError(message);
        if (!expired) toast.error(message);
        await refreshStatus();
      }
    } finally {
      generateAbort.current = null;
      setReloading(false);
      setPendingRun(null);
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
    model,
    generateAbort,
    willReload,
    toolRequest,
    overlapPanel,
    overlap,
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
    if (source) setGenerationError(null);
  }, [source]);

  return {
    source,
    sourceHandle,
    sourceStatus,
    setSourceStatus,
    sourceExpired,
    toolContext,
    toolPanels,
    toolValues,
    handleToolValueChange,
    family,
    overlap,
    willReload,
    reloading,
    pendingRun,
    estimateSeconds,
    cpuWarning,
    blocker,
    handleGenerate,
    generationError,
    setGenerationError,
    lastResult,
    clearLastResult: () => setLastResult(null),
  };
}

export type SeparateGeneration = ReturnType<typeof useSeparateGeneration>;
