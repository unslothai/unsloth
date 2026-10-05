// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { runAudio } from "../api";
import {
  type AudioOptionSpec,
  type AudioOptionValues,
  audioOptionLabel,
  audioOptionsForRequest,
  missingRequiredAudioOptions,
} from "../audio-options";
import { selectionExpired } from "../audio-run-request";
import type { AudioSourceInputHandle } from "../components/audio-source-input";
import { musicEditProblem } from "../components/music-edit-inputs";
import {
  buildMusicRunRequest,
  effectiveMusicMode,
  musicBlocker,
  reloadNotice,
  variationsFor,
} from "../music/music-policy";
import { parseMusicCapabilities } from "../music/music-types";
import type {
  MusicCapabilities,
  MusicMode,
  MusicModeRule,
} from "../music/music-types";
import type { GenerateBlocker } from "../pages/tts-workspace";
import { useAudioMusicStore } from "../stores/audio-music-store";
import type { AudioHostState } from "./audio-host-state";
import type { AudioSourceStatus } from "./audio-source-state";
import { REFERENCE_EXPIRED_MESSAGE } from "./audio-source-state";
import type { AudioGallery } from "./use-audio-gallery";
import type { AudioModelSlot } from "./use-audio-model-slot";
import { showRunResult } from "./use-clone-generation";

const MUSIC_PREVIEW_CAPABILITIES: MusicCapabilities = {
  modes: [
    {
      id: "song",
      lyrics: "optional",
      description: "optional",
      instrumental: "toggle",
      section_case: "lower",
      duration: { min: 5, max: 240, default: 30, approximate: false },
      variations: null,
    },
  ],
};

function requestedVariations(
  rule: MusicModeRule,
  song: { variations: number },
  sfx: { variations: number },
): number {
  if (rule.id === "song") return variationsFor(rule, song.variations);
  if (rule.id === "sfx") return variationsFor(rule, sfx.variations);
  return 1;
}

export function useMusicGeneration({
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
  lyrics,
  description,
  advancedOptionSpecs,
  audioOptionValues,
  toolOptions,
  toolBlocker,
  modelName,
  musicLoaded,
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
    lyrics: string;
    description: string;
    advancedOptionSpecs: AudioOptionSpec[];
    audioOptionValues: AudioOptionValues;
    toolOptions: AudioOptionValues | undefined;
    toolBlocker: string | null;
    modelName: string | null;
    musicLoaded: boolean;
  }) {
  const capabilities = useMemo(
    () =>
      parseMusicCapabilities(status?.audio_music) ??
      (musicLoaded ? null : MUSIC_PREVIEW_CAPABILITIES),
    [status?.audio_music, musicLoaded],
  );
  const pickedMode = useAudioMusicStore((state) => state.mode);
  const song = useAudioMusicStore((state) => state.song);
  const sfx = useAudioMusicStore((state) => state.sfx);
  const edit = useAudioMusicStore((state) => state.edit);
  const rule = capabilities
    ? effectiveMusicMode(capabilities, pickedMode)
    : null;
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<string | null>(null);
  const [sourceStatus, setSourceStatus] = useState<AudioSourceStatus>({
    phase: "idle",
  });
  const sourceHandle = useRef<AudioSourceInputHandle | null>(null);

  const editActions = useMemo(
    () => capabilities?.modes.find((mode) => mode.id === "edit")?.actions ?? [],
    [capabilities],
  );
  useEffect(() => {
    useAudioMusicStore.getState().setLoadedEditActions(editActions);
  }, [editActions]);

  const setMusicMode = useCallback((mode: MusicMode) => {
    useAudioMusicStore.getState().setMode(mode);
  }, []);

  const variations = rule ? requestedVariations(rule, song, sfx) : 1;
  const notice = rule ? reloadNotice(rule, variations, modelName) : null;

  const sourceExpired =
    sourceStatus.phase === "expired" ||
    selectionExpired(edit.source, Date.now());
  const editProblem =
    rule?.id === "edit"
      ? sourceStatus.phase === "uploading" || sourceStatus.phase === "recording"
        ? "Waiting for the clip to finish uploading."
        : sourceStatus.phase === "error"
          ? sourceStatus.message
          : sourceExpired
            ? REFERENCE_EXPIRED_MESSAGE
            : musicEditProblem(rule, edit, edit.source?.durationS ?? null)
      : null;
  const pageProblem = rule
    ? musicBlocker({ rule, description, lyrics, song, sfx, edit, editProblem })
    : null;
  const blocker: GenerateBlocker | null = !rule
    ? null
    : pageProblem
      ? {
          reason: pageProblem,
          ...(rule.id === "edit" && !edit.source
            ? {
                actions: [
                  {
                    label: "Add a clip",
                    onClick: () => sourceHandle.current?.browse(),
                  },
                ],
              }
            : {}),
        }
      : toolBlocker
        ? { reason: toolBlocker }
        : null;

  const handleGenerate = useCallback(async () => {
    if (!rule || blocker) return;
    if (busyRef.current) return;
    busyRef.current = "generating";
    setBusy("generating");
    updateGenerationPhase("preparing");
    const releaseInFlight = pendingTranscribeRelease.current;
    if (releaseInFlight && !(await releaseInFlight)) {
      updateGenerationPhase(null);
      busyRef.current = null;
      setBusy(null);
      setMode("transcribe");
      return;
    }
    const missingOptions = missingRequiredAudioOptions(
      advancedOptionSpecs,
      audioOptionValues,
    );
    if (missingOptions.length > 0) {
      updateGenerationPhase(null);
      busyRef.current = null;
      setBusy(null);
      setAdvancedOpen(true);
      toast.error(
        `Set ${missingOptions.map((spec) => audioOptionLabel(spec.name)).join(", ")} in Advanced before generating.`,
      );
      return;
    }
    const request = buildMusicRunRequest({
      rule,
      description,
      lyrics,
      song,
      sfx,
      edit,
      // Panel options override Advanced: panels own their options.
      options: {
        ...audioOptionsForRequest(advancedOptionSpecs, audioOptionValues),
        ...toolOptions,
      },
    });
    const reloads = notice !== null;
    setRunStatus(
      reloads
        ? `Reloading ${modelName ?? "the model"} for ${variations} variations…`
        : rule.id === "edit"
          ? "Editing the clip…"
          : variations > 1
            ? `Making ${variations} variations…`
            : rule.id === "sfx"
              ? "Making the sound…"
              : "Making the music…",
    );
    const controller = new AbortController();
    generateAbort.current = controller;
    updateGenerationPhase("generating");
    try {
      const response = await runAudio(request, controller.signal);
      updateGenerationPhase("finishing");
      await showRunResult({
        response,
        text: request.text || request.music?.lyrics || "",
        workflow: "music",
        refreshGallery,
        selectClip,
        setFallbackClip,
        setSelectedId,
      });
      if (reloads) await refreshStatus();
    } catch (error) {
      if (!controller.signal.aborted) {
        updateGenerationPhase("finishing");
        const message =
          error instanceof Error ? error.message : "Music generation failed.";
        setGenerationError(message);
        toast.error(message);
        await refreshStatus();
      }
    } finally {
      setRunStatus(null);
      generateAbort.current = null;
      updateGenerationPhase(null);
      busyRef.current = null;
      setBusy(null);
      if (activeRef.current && modeRef.current === "speak")
        replayQueuedTtsPick();
    }
  }, [
    rule,
    blocker,
    busyRef,
    setBusy,
    updateGenerationPhase,
    pendingTranscribeRelease,
    setMode,
    advancedOptionSpecs,
    audioOptionValues,
    setAdvancedOpen,
    description,
    lyrics,
    song,
    sfx,
    edit,
    toolOptions,
    notice,
    modelName,
    variations,
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

  return {
    studio: capabilities !== null,
    capabilities,
    rule,
    setMusicMode,
    song,
    sfx,
    edit,
    variations,
    reloadNotice: notice,
    blocker,
    handleGenerate,
    generationError,
    setGenerationError,
    runStatus,
    setSourceStatus,
    sourceHandle,
  };
}

export type MusicGeneration = ReturnType<typeof useMusicGeneration>;
