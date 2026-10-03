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
  parseAudioOptions,
} from "../audio-options";
import {
  type AudioSourceSelection,
  selectionExpired,
  sourceRefOf,
} from "../audio-run-request";
import type { AudioSourceInputHandle } from "../components/audio-source-input";
import {
  type ConvertBlockerKind,
  convertBlocker,
  convertCaps,
  convertPitchSupport,
  convertServerTask,
  convertSwitchNotice,
  effectiveConvertStyle,
} from "../convert-policy";
import type { GenerateBlocker } from "../pages/tts-workspace";
import { useAudioCloneStore } from "../stores/audio-clone-store";
import { useAudioConvertStore } from "../stores/audio-convert-store";
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
import type { AudioModelSlot } from "./use-audio-model-slot";
import { showRunResult } from "./use-clone-generation";
import { useReferenceTranscribe } from "./use-reference-transcribe";

export const CONVERT_SOURCE_TEXT_FIELD_ID = "convert-source-text";

// Seed-VC starts on its V2 engine until a run picks another one.
const SEED_VC_DEFAULT_ROUTE = "v2_vc";

const IDLE: AudioSourceStatus = { phase: "idle" };

function sourceBusy(status: AudioSourceStatus): boolean {
  return status.phase === "uploading" || status.phase === "recording";
}

export function useConvertGeneration({
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
  const source = useAudioConvertStore((state) => state.source);
  const target = useAudioConvertStore((state) => state.target);
  const builtinVoice = useAudioConvertStore((state) => state.builtinVoice);
  const storedMode = useAudioConvertStore((state) => state.mode);
  const pitchAuto = useAudioConvertStore((state) => state.pitchAuto);
  const pitch = useAudioConvertStore((state) => state.pitch);
  const sourceText = useAudioConvertStore((state) => state.sourceText);
  const storedToolValues = useAudioCloneStore((state) => state.toolValues);
  const model = status?.active_model ?? null;
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [sourceStatus, setSourceStatus] = useState<AudioSourceStatus>(IDLE);
  const [targetStatus, setTargetStatus] = useState<AudioSourceStatus>(IDLE);
  const sourceHandle = useRef<AudioSourceInputHandle | null>(null);
  const targetHandle = useRef<AudioSourceInputHandle | null>(null);
  const [expiredIds, setExpiredIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  // Uploading the same audio again returns the same id, which is live again.
  const sourceId = source?.id ?? null;
  const targetId = target?.id ?? null;
  useEffect(() => {
    setExpiredIds((previous) => {
      if (![sourceId, targetId].some((id) => id && previous.has(id))) {
        return previous;
      }
      const next = new Set(previous);
      for (const id of [sourceId, targetId]) {
        if (id) {
          next.delete(id);
        }
      }
      return next;
    });
  }, [sourceId, targetId]);
  const [runningNotice, setRunningNotice] = useState<string | null>(null);

  const transcriber = useReferenceTranscribe({
    sttRepo,
    language: "",
    purpose: "convert",
    onText: (next) => useAudioConvertStore.getState().setSourceText(next),
  });
  // A transcription still running for the previous recording must not fill in the new one's text.
  const sourceKey = source ? `${source.kind}:${source.id}` : null;
  const cancelTranscribe = transcriber.cancel;
  useEffect(() => cancelTranscribe(), [sourceKey, cancelTranscribe]);

  const toolContext = useMemo(
    () =>
      audioModelContextFor(status, {
        musicGeneration: false,
        cudaMusicGeneration: false,
        musicNeedsDescription: false,
      }),
    [status],
  );
  const caps = convertCaps(toolContext);
  const mode =
    caps && !caps.modes.includes(storedMode)
      ? (caps.modes[0] ?? "speech")
      : storedMode;
  const toolPanels = useMemo(
    () => audioToolPanelsFor("convert", toolContext),
    [toolContext],
  );
  const toolValues = useMemo(() => {
    const values: Record<string, unknown> = {};
    for (const panel of toolPanels) {
      values[panel.id] =
        storedToolValues[toolValueKey(model, "convert", panel.id)];
    }
    return values;
  }, [toolPanels, storedToolValues, model]);
  const handleToolValueChange = useCallback(
    (panelId: string, value: unknown) =>
      useAudioCloneStore
        .getState()
        .setToolValue(toolValueKey(model, "convert", panelId), value),
    [model],
  );
  const convertOptionSpecs = useMemo(() => {
    const byWorkflow = status?.audio_options_by_workflow?.convert;
    return byWorkflow === undefined
      ? audioOptionSpecs
      : parseAudioOptions(byWorkflow);
  }, [status?.audio_options_by_workflow, audioOptionSpecs]);
  const toolRequest = useMemo(
    () =>
      collectToolRequest(
        toolPanels,
        toolValues,
        { text: "", referenceText: sourceText, hasReference: source !== null },
        { ...toolContext, convertMode: mode },
        convertOptionSpecs,
      ),
    [
      toolPanels,
      toolValues,
      sourceText,
      source,
      toolContext,
      mode,
      convertOptionSpecs,
    ],
  );
  const style = effectiveConvertStyle(
    caps,
    mode,
    toolRequest.patch.convert?.style ?? "source",
  );
  const route = toolRequest.patch.convert?.route ?? null;
  const pitchSupport = convertPitchSupport(caps, mode, style);
  const claimedOptions = useMemo(
    () => claimedOptionNames(toolPanels),
    [toolPanels],
  );
  const advancedOptionSpecs = useMemo(
    () => convertOptionSpecs.filter((spec) => !claimedOptions.has(spec.name)),
    [convertOptionSpecs, claimedOptions],
  );

  const nextTask = convertServerTask(caps, status?.audio_workflow_tasks, mode);
  const loadedTask = status?.audio_server_task ?? null;
  const loadedRoute =
    status?.audio_convert_route ??
    (loadedTask === nextTask ? SEED_VC_DEFAULT_ROUTE : null);
  const switchNotice =
    caps && model
      ? convertSwitchNotice({
          modelName: model.split("/").pop() ?? model,
          family: status?.audio_family ?? null,
          loadedTask,
          nextTask,
          routeChange:
            caps.route_reloads &&
            route !== null &&
            loadedRoute !== null &&
            route !== loadedRoute,
        })
      : null;

  // An upload can pass its keep-until time while the page sits open.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const expiring = [source, target].some(
      (item) => item?.kind === "input" && item.expiresAt,
    );
    if (!expiring) return;
    const timer = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, [source, target]);
  const isExpired = (
    selection: AudioSourceSelection | null,
    cardStatus: AudioSourceStatus,
  ) =>
    cardStatus.phase === "expired" ||
    selectionExpired(selection, now) ||
    (selection !== null && expiredIds.has(selection.id));
  const sourceExpired = isExpired(source, sourceStatus);
  const targetExpired = isExpired(target, targetStatus);
  const builtinTarget = caps?.target === "builtin";

  const inputBlocker = convertBlocker({
    source,
    sourceBusy: sourceBusy(sourceStatus),
    sourceExpired,
    sourceError: sourceStatus.phase === "error" ? sourceStatus.message : null,
    target: builtinTarget ? null : target,
    targetBusy: sourceBusy(targetStatus),
    targetExpired: !builtinTarget && targetExpired,
    targetError:
      !builtinTarget && targetStatus.phase === "error"
        ? targetStatus.message
        : null,
    builtinVoice,
    caps,
    mode,
    style,
    sourceText,
    panelError: toolRequest.error,
  });
  const againAction = (
    handle: typeof sourceHandle,
    clear: () => void,
  ): GenerateBlocker["actions"] => [
    {
      label: "Add it again",
      onClick: () => {
        clear();
        handle.current?.browse();
      },
    },
  ];
  const focusAction = (label: string, handle: typeof sourceHandle) => [
    { label, onClick: () => handle.current?.focus() },
  ];
  const blockerActions = (
    kind: ConvertBlockerKind,
  ): GenerateBlocker["actions"] => {
    switch (kind) {
      case "source":
      case "source-error":
        return focusAction("Add a recording", sourceHandle);
      case "target":
      case "target-error":
        return focusAction("Add the target voice", targetHandle);
      case "source-expired":
        return againAction(sourceHandle, () =>
          useAudioConvertStore.getState().setSource(null),
        );
      case "target-expired":
        return againAction(targetHandle, () =>
          useAudioConvertStore.getState().setTarget(null),
        );
      case "source-text":
        return [
          {
            label: "Transcribe it",
            onClick: () => void transcriber.transcribe(source),
          },
          {
            label: "type it",
            onClick: () =>
              document.getElementById(CONVERT_SOURCE_TEXT_FIELD_ID)?.focus(),
          },
        ];
      default:
        return undefined;
    }
  };
  const blocker: GenerateBlocker | null = inputBlocker
    ? {
        reason: inputBlocker.reason,
        actions: blockerActions(inputBlocker.kind),
      }
    : null;

  const handleGenerate = useCallback(async () => {
    const state = useAudioConvertStore.getState();
    if (!state.source || inputBlocker || !caps) return;
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
        `Set ${missingOptions.map((spec) => audioOptionLabel(spec.name)).join(", ")} in Advanced before converting.`,
      );
      return;
    }
    const patch = toolRequest.patch;
    const sourceSelection = state.source;
    const targetSelection = builtinTarget ? null : state.target;
    const controller = new AbortController();
    generateAbort.current = controller;
    setRunningNotice(switchNotice?.during ?? null);
    updateGenerationPhase(switchNotice ? "switching" : "generating");
    const autoPitch = pitchSupport.show && pitchSupport.auto && state.pitchAuto;
    try {
      const response = await runAudio(
        {
          workflow: "convert",
          inputs: {
            source: sourceRefOf(sourceSelection),
            target: targetSelection ? sourceRefOf(targetSelection) : null,
            source_text: style === "target" ? state.sourceText.trim() : null,
          },
          convert: {
            mode,
            pitch: pitchSupport.show && !autoPitch ? state.pitch : null,
            pitch_auto: autoPitch,
            style: caps.style ? style : undefined,
            voice: builtinTarget ? state.builtinVoice : null,
          },
          options: {
            ...audioOptionsForRequest(advancedOptionSpecs, audioOptionValues),
            ...patch.options,
          },
        },
        controller.signal,
      );
      updateGenerationPhase("finishing");
      await showRunResult({
        response,
        text: `${sourceSelection.name} → ${
          targetSelection?.name ?? state.builtinVoice
        }`,
        refreshGallery,
        selectClip,
        setFallbackClip,
        setSelectedId,
      });
      if (switchNotice) await refreshStatus();
    } catch (error) {
      if (!controller.signal.aborted) {
        updateGenerationPhase("finishing");
        // Only the expired-upload 404: a deleted clip or voice says otherwise and leaves uploads alone.
        const gone =
          error instanceof AudioApiError &&
          error.status === 404 &&
          error.message === REFERENCE_EXPIRED_MESSAGE
            ? [sourceSelection, targetSelection].filter(
                (item): item is AudioSourceSelection => item?.kind === "input",
              )
            : [];
        if (gone.length > 0) {
          setExpiredIds((previous) => {
            const next = new Set(previous);
            for (const item of gone) next.add(item.id);
            return next;
          });
          for (const item of gone) {
            (item === sourceSelection
              ? sourceHandle
              : targetHandle
            ).current?.markExpired();
          }
        }
        const message =
          gone.length > 0
            ? REFERENCE_EXPIRED_MESSAGE
            : error instanceof Error
              ? error.message
              : "Voice conversion failed.";
        setGenerationError(message);
        if (gone.length === 0) toast.error(message);
        await refreshStatus();
      } else if (switchNotice) {
        // Stopped while switching: the server may or may not run the new task now.
        await refreshStatus();
      }
    } finally {
      generateAbort.current = null;
      setRunningNotice(null);
      updateGenerationPhase(null);
      busyRef.current = null;
      setBusy(null);
      if (activeRef.current && modeRef.current === "speak")
        replayQueuedTtsPick();
    }
  }, [
    inputBlocker,
    caps,
    busyRef,
    setBusy,
    updateGenerationPhase,
    pendingTranscribeRelease,
    setMode,
    advancedOptionSpecs,
    audioOptionValues,
    setAdvancedOpen,
    toolRequest,
    builtinTarget,
    generateAbort,
    switchNotice,
    pitchSupport,
    style,
    mode,
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
    if (source || target) setGenerationError(null);
  }, [source, target]);

  return {
    source,
    target,
    builtinVoice,
    mode,
    style,
    pitch,
    pitchAuto,
    pitchSupport,
    sourceText,
    caps,
    sourceHandle,
    targetHandle,
    sourceStatus,
    setSourceStatus,
    targetStatus,
    setTargetStatus,
    sourceExpired,
    targetExpired,
    transcriber,
    toolContext,
    toolPanels,
    toolValues,
    handleToolValueChange,
    claimedOptions,
    advancedOptionSpecs,
    switchNotice,
    runningNotice,
    blocker,
    inputsReady: inputBlocker === null,
    handleGenerate,
    generationError,
    setGenerationError,
  };
}

export type ConvertGeneration = ReturnType<typeof useConvertGeneration>;
