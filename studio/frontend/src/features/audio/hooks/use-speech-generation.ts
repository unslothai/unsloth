// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useState } from "react";
import { readLastPrompt, saveLastPrompt } from "@/lib/last-prompt";
import { toast } from "@/lib/toast";
import { generateAudio, runAudio } from "../api";
import { TTS_MAX_TOKENS } from "../audio-workspace-constants";
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

/** Where a page's draft is kept. Speak's text uses the key the single Audio page always used. */
function ttsDraftKey(field: TtsDraftField, workflow: TtsWorkflow): string {
  const base = workflow === "music" ? "audio:music" : "audio";
  return field === "prompt" ? base : `${base}:instructions`;
}

/** Speak and Music generation: the text and settings, what the loaded model needs, and the generate request. */
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
  // Speak and Music each keep their own draft of the text and the description, so switching pages
  // never overwrites the other one, and both survive a reload. Speak's text key predates the split.
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
  // The page is read at call time, as on Images, so a stale closure cannot write the other page's draft.
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
  // Sending temperature unconditionally puts it in the request's model_fields_set, which the backend reads as an
  // explicit client override that beats the per-model recommendation (Spark-TTS wants 0.8, OuteTTS 0.4), so only
  // send it once the user has moved the slider.
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
  // Most GGUF music falls back to the lyrics as its prompt; MiniMax Music 3 and YuE2 need a
  // description beside them.
  const musicNeedsDescription =
    cudaMusicGeneration ||
    musicModelNeedsDescription(status?.audio_type, status?.audio_family);
  const lyricsOptional = musicLyricsOptional(
    status?.audio_type,
    status?.audio_family,
  );
  const musicRange = musicDurationRange(cudaMusicGeneration);
  // A length picked for the MiniMax pipeline can exceed what the GGUF runtime generates.
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
  // Each model keeps its own values, read back when it becomes the loaded one.
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

  // The model tools this page shows for the loaded model. The instruction panels keep editing the
  // page's instruction draft, as the rail always did; every other panel keeps its value per model.
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
  /** Spec options a shown panel renders itself, which Advanced leaves out. */
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
    // Same gate the TTS load path uses: switching straight from Transcribe with a speech model already resident
    // needs no load, so nothing else waits for the sidecar teardown, and generating beside a dictation model OOMs a
    // device that fits either alone. Claimed before the await below, since the button only disables on `busy` and a
    // slow release let several clicks each resume into their own generateAudio.
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
    // Advanced first, then what the shown panels set, which own their options.
    const requestOptions = {
      ...audioOptionsForRequest(advancedOptionSpecs, audioOptionValues),
      ...toolRequest.patch.options,
    };
    const savedVoice =
      workflow === "speak" ? toolRequest.patch.inputs?.reference : undefined;
    const language = audioLanguage.trim();
    saveLastPrompt(ttsDraftKey("prompt", workflow), prompt);
    if (savedVoice) {
      // A saved voice is a reference the server holds, so the run goes through /audio/run.
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
        // The server did persist it; only this refresh missed it. Select the id so a later refresh shows the real
        // record, but keep the response audio too: selectedClip resolves against `clips`, so an id that is not there
        // yet would render the empty state.
        setFallbackClip({
          url: `data:audio/wav;base64,${generated.audio.data}`,
          prompt: text,
          model: generated.model,
          saved: true,
        });
        selectClip(generated.clip_id, true);
      } else {
        // Gallery persistence is best-effort server-side, so a full disk still returns the audio. Play it
        // from the response rather than dropping an expensive generation.
        galleryCache.selectedId = null;
        setSelectedId(null);
        setFallbackClip({
          url: `data:audio/wav;base64,${generated.audio.data}`,
          prompt: text,
          model: generated.model,
          saved: false,
        });
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        updateGenerationPhase("finishing");
        // Kept under Generate too, so the reason outlives the toast.
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

  // Only unmount aborts. RootLayout keeps this page mounted precisely so leaving the tab does not
  // cancel synthesis, and the clip is persisted server-side.
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
    /** Why a shown panel holds Generate back, in words for the blocker line. */
    toolBlocker: toolRequest.error,
    claimedOptions,
    advancedOptionSpecs,
  };
}

export type SpeechGeneration = ReturnType<typeof useSpeechGeneration>;
