// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useState } from "react";
import { readLastPrompt, saveLastPrompt } from "@/lib/last-prompt";
import { toast } from "@/lib/toast";
import { generateAudio, runAudio } from "../api";
import {
  audioOptionLabel,
  audioOptionsForRequest,
  type AudioOptionValue,
  type AudioOptionValues,
  missingRequiredAudioOptions,
  parseAudioOptions,
  readAudioOptionValues,
  saveAudioOptionValues,
} from "../audio-options";
import {
  audioSamplingControlsApply,
  isTtsAudioType,
  MINIMAX_MUSIC_DEFAULT_SECONDS,
  minimaxMusicFramesForSeconds,
  MOSS_TTS_DEFAULT_SECONDS,
  MOSS_TTS_FRAMES_PER_SECOND,
  MOSS_TTS_MAX_FRAMES,
  mossTtsFramesForSeconds,
  mossTtsMaxFrames,
  musicDurationRange,
  musicLyricsOptional,
  musicNeedsDescription as musicModelNeedsDescription,
  nativeAudioInstructionsKind,
  persistedClipForGeneration,
} from "../audio-page-policy";
import {
  isMusicGenerationModel,
  musicGenerationRequiresCuda,
} from "../catalog";
import { TTS_MAX_TOKENS } from "../audio-workspace-constants";
import type { AudioHostState } from "./audio-host-state";
import { galleryCache } from "./use-audio-gallery";
import { useAudioCloneStore } from "../stores/audio-clone-store";
import { useAudioWorkspaceStore } from "../stores/audio-workspace-store";
import { audioToolPanelsFor, isInstructionPanel } from "../tools/registry";
import {
  audioModelContextFor,
  claimedOptionNames,
  collectToolRequest,
  toolValueKey,
} from "../tools/select";
import { showRunResult } from "./use-clone-generation";
import type { AudioWorkflowId } from "../workflows";
import type { AudioGallery } from "./use-audio-gallery";
import type { AudioModelSlot } from "./use-audio-model-slot";

type TtsWorkflow = "speak" | "music";
type TtsDraftField = "prompt" | "instructions";

function ttsWorkflowOf(workflow: AudioWorkflowId): TtsWorkflow {
  return workflow === "music" ? "music" : "speak";
}

/** Speak's text keeps the pre-split key so existing drafts survive. */
function ttsDraftKey(field: TtsDraftField, workflow: TtsWorkflow): string {
  const base = workflow === "music" ? "audio:music" : "audio";
  return field === "prompt" ? base : `${base}:instructions`;
}

export function useSpeechGeneration({
  workflow,
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
}: { workflow: TtsWorkflow } & Pick<
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
  Pick<AudioGallery, "refreshGallery" | "selectClip" | "setFallbackClip" | "setSelectedId"> &
  Pick<AudioModelSlot, "pendingTranscribeRelease" | "replayQueuedTtsPick">) {
  const [drafts, setDrafts] = useState<Record<TtsDraftField, Record<TtsWorkflow, string>>>(
    () => ({
      prompt: {
        speak: readLastPrompt(ttsDraftKey("prompt", "speak")),
        music: readLastPrompt(ttsDraftKey("prompt", "music")),
      },
      instructions: {
        speak: readLastPrompt(ttsDraftKey("instructions", "speak")),
        music: readLastPrompt(ttsDraftKey("instructions", "music")),
      },
    }),
  );
  const [generationError, setGenerationError] = useState<string | null>(null);
  const prompt = drafts.prompt[workflow];
  const audioInstructions = drafts.instructions[workflow];
  const setDraft = useCallback((field: TtsDraftField, next: string) => {
    const page = ttsWorkflowOf(useAudioWorkspaceStore.getState().workflow);
    saveLastPrompt(ttsDraftKey(field, page), next);
    setDrafts((current) => ({
      ...current,
      [field]: { ...current[field], [page]: next },
    }));
  }, []);
  const setPrompt = useCallback(
    (next: string) => setDraft("prompt", next),
    [setDraft],
  );
  const setAudioInstructions = useCallback(
    (next: string) => setDraft("instructions", next),
    [setDraft],
  );
  const [audioLanguage, setAudioLanguage] = useState("");
  const [temperature, setTemperature] = useState(0.6);
  // Only send temperature once moved: model_fields_set makes it override the per-model recommendation.
  const [temperatureEdited, setTemperatureEdited] = useState(false);
  const handleTemperatureChange = useCallback((value: number) => {
    setTemperatureEdited(true);
    setTemperature(value);
  }, []);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [mossMaxSeconds, setMossMaxSeconds] = useState(
    MOSS_TTS_DEFAULT_SECONDS,
  );
  const [minimaxMaxSeconds, setMinimaxMaxSeconds] = useState(
    MINIMAX_MUSIC_DEFAULT_SECONDS,
  );
  const mossFrameLimit = mossTtsMaxFrames(
    status?.audio_type,
    status?.context_length,
  );
  const mossMaxSecondsLimit =
    (mossFrameLimit ?? MOSS_TTS_MAX_FRAMES) / MOSS_TTS_FRAMES_PER_SECOND;

  useEffect(() => {
    setTemperature(mossFrameLimit !== null ? 1.7 : 0.6);
    setTemperatureEdited(false);
    if (mossFrameLimit !== null) {
      setMossMaxSeconds((current) =>
        Math.min(current, mossFrameLimit / MOSS_TTS_FRAMES_PER_SECOND),
      );
    }
  }, [mossFrameLimit]);

  const ttsLoaded = Boolean(
    status?.active_model &&
      isTtsAudioType(status.audio_type, status.is_gguf === true),
  );
  const musicGeneration = isMusicGenerationModel(
    status?.active_model,
    status?.audio_type,
  );
  const cudaMusicGeneration = musicGenerationRequiresCuda(
    status?.active_model,
    status?.audio_type,
  );
  const musicNeedsDescription =
    cudaMusicGeneration ||
    musicModelNeedsDescription(status?.audio_type, status?.audio_family);
  const lyricsOptional = musicLyricsOptional(
    status?.audio_type,
    status?.audio_family,
  );
  const musicRange = musicDurationRange(cudaMusicGeneration);
  const musicSeconds = Math.min(
    Math.max(minimaxMaxSeconds, musicRange.min),
    musicRange.max,
  );
  const samplingControls = audioSamplingControlsApply(status?.audio_type);
  const audioOptionSpecs = useMemo(
    () => parseAudioOptions(status?.audio_options),
    [status?.audio_options],
  );
  const audioOptionsModel = status?.active_model ?? null;
  const [audioOptionValues, setAudioOptionValues] = useState<AudioOptionValues>(
    () => readAudioOptionValues(audioOptionsModel),
  );
  const [audioOptionValuesModel, setAudioOptionValuesModel] = useState(
    audioOptionsModel,
  );
  if (audioOptionValuesModel !== audioOptionsModel) {
    setAudioOptionValuesModel(audioOptionsModel);
    setAudioOptionValues(readAudioOptionValues(audioOptionsModel));
  }
  const handleAudioOptionChange = useCallback(
    (name: string, value: AudioOptionValue | undefined) => {
      setAudioOptionValues((current) => {
        const next = { ...current };
        if (value === undefined) delete next[name];
        else next[name] = value;
        saveAudioOptionValues(audioOptionsModel, next);
        return next;
      });
    },
    [audioOptionsModel],
  );
  const handleAudioOptionsReset = useCallback(() => {
    setAudioOptionValues({});
    saveAudioOptionValues(audioOptionsModel, {});
  }, [audioOptionsModel]);
  const mossLocalGeneration = status?.audio_type === "moss_tts_local";
  const instructionsKind = musicGeneration
    ? "music"
    : nativeAudioInstructionsKind(status?.audio_type);

  // Instruction panels edit the page's instruction draft; other panels keep a value per model.
  const toolContext = useMemo(
    () =>
      audioModelContextFor(status, {
        musicGeneration: workflow === "music",
        cudaMusicGeneration,
        musicNeedsDescription,
      }),
    [status, workflow, cudaMusicGeneration, musicNeedsDescription],
  );
  const toolPanels = useMemo(
    () => audioToolPanelsFor(workflow, toolContext),
    [workflow, toolContext],
  );
  const storedToolValues = useAudioCloneStore((state) => state.toolValues);
  const toolValues = useMemo(() => {
    const values: Record<string, unknown> = {};
    for (const panel of toolPanels) {
      values[panel.id] = isInstructionPanel(panel.id)
        ? { instructions: audioInstructions, language: audioLanguage }
        : storedToolValues[toolValueKey(audioOptionsModel, workflow, panel.id)];
    }
    return values;
  }, [
    toolPanels,
    audioInstructions,
    audioLanguage,
    storedToolValues,
    audioOptionsModel,
    workflow,
  ]);
  const handleToolValueChange = useCallback(
    (panelId: string, value: unknown) => {
      if (isInstructionPanel(panelId)) {
        const next = value as { instructions?: string; language?: string };
        if (typeof next.instructions === "string")
          setAudioInstructions(next.instructions);
        if (typeof next.language === "string") setAudioLanguage(next.language);
        return;
      }
      useAudioCloneStore
        .getState()
        .setToolValue(
          toolValueKey(audioOptionsModel, workflow, panelId),
          value,
        );
    },
    [setAudioInstructions, audioOptionsModel, workflow],
  );
  const toolRequest = useMemo(
    () =>
      collectToolRequest(
        toolPanels,
        toolValues,
        { text: prompt },
        toolContext,
        audioOptionSpecs,
      ),
    [toolPanels, toolValues, prompt, toolContext, audioOptionSpecs],
  );
  const claimedOptions = useMemo(
    () => claimedOptionNames(toolPanels),
    [toolPanels],
  );
  const advancedOptionSpecs = useMemo(
    () => audioOptionSpecs.filter((spec) => !claimedOptions.has(spec.name)),
    [audioOptionSpecs, claimedOptions],
  );

  const handleGenerate = useCallback(async () => {
    const text = prompt.trim();
    if (!text && !lyricsOptional) return;
    // Same sidecar gate as the TTS load path, claimed before the await: generating beside a dictation model OOMs.
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
    const instructions = audioInstructions.trim();
    if (musicNeedsDescription && !instructions) {
      updateGenerationPhase(null);
      busyRef.current = null;
      setBusy(null);
      toast.error(
        "Add a music description. This model needs one beside the lyrics.",
      );
      return;
    }
    if (toolRequest.error) {
      updateGenerationPhase(null);
      busyRef.current = null;
      setBusy(null);
      toast.error(toolRequest.error);
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
    const requestOptions = {
      ...audioOptionsForRequest(advancedOptionSpecs, audioOptionValues),
      ...toolRequest.patch.options,
    };
    const savedVoice =
      workflow === "speak" ? toolRequest.patch.inputs?.reference : undefined;
    const language = audioLanguage.trim();
    saveLastPrompt(ttsDraftKey("prompt", workflow), prompt);
    if (savedVoice) {
      const voiceController = new AbortController();
      generateAbort.current = voiceController;
      updateGenerationPhase("generating");
      try {
        const response = await runAudio(
          {
            workflow: "speak",
            text,
            instructions:
              instructionsKind !== null && instructions ? instructions : null,
            language: mossLocalGeneration && language ? language : null,
            inputs: { reference: savedVoice },
            options: requestOptions,
            speed: toolRequest.patch.speed ?? null,
          },
          voiceController.signal,
        );
        updateGenerationPhase("finishing");
        await showRunResult({
          response,
          text,
          workflow: "speak",
          refreshGallery,
          selectClip,
          setFallbackClip,
          setSelectedId,
        });
      } catch (error) {
        if (!voiceController.signal.aborted) {
          updateGenerationPhase("finishing");
          const message =
            error instanceof Error ? error.message : "Audio generation failed.";
          setGenerationError(message);
          toast.error(message);
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
      return;
    }
    const controller = new AbortController();
    generateAbort.current = controller;
    updateGenerationPhase("generating");
    try {
      const generated = await generateAudio(text, {
        ...(!musicGeneration && samplingControls && temperatureEdited
          ? { temperature }
          : {}),
        max_tokens: musicGeneration
          ? minimaxMusicFramesForSeconds(musicSeconds)
          : mossFrameLimit !== null
            ? mossTtsFramesForSeconds(mossMaxSeconds, mossFrameLimit)
            : maxTokens,
        ...(instructionsKind !== null && instructions
          ? { audio_instructions: instructions }
          : {}),
        ...(mossLocalGeneration && language
          ? { audio_language: language }
          : {}),
        ...(Object.keys(requestOptions).length > 0
          ? { audio_options: requestOptions }
          : {}),
        signal: controller.signal,
      });
      if (generated.choices[0]?.finish_reason === "length")
        toast.warning(
          maxTokens < TTS_MAX_TOKENS
            ? "Speech stopped at the Max tokens limit before the end of the text. Raise Max tokens under Advanced to hear the rest."
            : "Speech stopped at the Max tokens limit before the end of the text. Split the text into shorter parts to hear the rest.",
        );
      updateGenerationPhase("finishing");
      const refreshed = await refreshGallery();
      const generatedClip = persistedClipForGeneration(
        generated.clip_id,
        refreshed,
      );
      if (generatedClip) {
        setFallbackClip(null);
        selectClip(generatedClip.id);
      } else if (generated.clip_id) {
        // Persisted but missed by this refresh: keep the response audio, or selectedClip renders empty.
        setFallbackClip({
          url: `data:audio/wav;base64,${generated.audio.data}`,
          prompt: text,
          model: generated.model,
          saved: true,
          workflow,
        });
        selectClip(generated.clip_id, true);
      } else {
        galleryCache.selectedId = null;
        setSelectedId(null);
        setFallbackClip({
          url: `data:audio/wav;base64,${generated.audio.data}`,
          prompt: text,
          model: generated.model,
          saved: false,
          workflow,
        });
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        updateGenerationPhase("finishing");
        setGenerationError(
          error instanceof Error ? error.message : "Audio generation failed.",
        );
        toast.error(
          error instanceof Error ? error.message : "Audio generation failed.",
        );
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
    workflow,
    prompt,
    audioInstructions,
    audioLanguage,
    musicGeneration,
    musicNeedsDescription,
    lyricsOptional,
    advancedOptionSpecs,
    audioOptionValues,
    toolRequest,
    setAdvancedOpen,
    mossLocalGeneration,
    mossFrameLimit,
    mossMaxSeconds,
    musicSeconds,
    samplingControls,
    instructionsKind,
    temperature,
    temperatureEdited,
    updateGenerationPhase,
    maxTokens,
    refreshGallery,
    refreshStatus,
    replayQueuedTtsPick,
    selectClip,
  ]);

  // Only unmount aborts: RootLayout keeps this page mounted so leaving the tab does not cancel synthesis.
  useEffect(() => () => generateAbort.current?.abort(), []);

  return {
    prompt,
    setPrompt,
    audioInstructions,
    setAudioInstructions,
    audioLanguage,
    setAudioLanguage,
    temperature,
    handleTemperatureChange,
    maxTokens,
    setMaxTokens,
    mossMaxSeconds,
    setMossMaxSeconds,
    setMinimaxMaxSeconds,
    mossFrameLimit,
    mossMaxSecondsLimit,
    ttsLoaded,
    musicGeneration,
    cudaMusicGeneration,
    musicNeedsDescription,
    lyricsOptional,
    musicRange,
    musicSeconds,
    samplingControls,
    audioOptionSpecs,
    audioOptionValues,
    handleAudioOptionChange,
    handleAudioOptionsReset,
    mossLocalGeneration,
    instructionsKind,
    handleGenerate,
    generationError,
    setGenerationError,
    toolContext,
    toolPanels,
    toolValues,
    handleToolValueChange,
    toolBlocker: toolRequest.error,
    claimedOptions,
    advancedOptionSpecs,
  };
}

export type SpeechGeneration = ReturnType<typeof useSpeechGeneration>;
