// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Audio page: Speak and Music (the main inference slot) and Transcribe (STT via the dictation sidecar).
// It hosts the shared shell and state; the hooks own each concern and pages/ the rail and output of each
// workflow. The page stays mounted across tab switches (see __root.tsx), so `active` gates polling, popovers
// and the recorder rather than lifecycle.

import { TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import { SparklesIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { LibraryPageLink } from "@/components/media-page-link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { MediaRailResizeHandle } from "@/components/media-rail-resize-handle";
import { MEDIA_RAIL_ROOT_ATTR, useMediaRailWidth } from "@/hooks/use-media-rail-width";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { buildAudioTourSteps } from "./tour";
import { useSidebar } from "@/components/ui/sidebar";
import { usePlatformStore } from "@/config/env";
import { type InferenceStatusResponse, getInferenceStatus } from "@/features/chat";
import { ModelSelector } from "@/features/model-picker/components/model-selector";
import { AUDIO_CATALOG } from "@/features/model-picker/components/model-selector/model-catalog";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import type { ModelOption } from "@/features/model-picker/components/model-selector/types";
import { sttModelSize } from "@/features/settings/stores/stt-model-catalog";
import { usePersistedChoice } from "@/hooks/use-persisted-choice";
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
import { useScrollFades } from "@/hooks/use-scroll-fades";
import { isTauri } from "@/lib/api-base";
import { subscribeModelLifecycle } from "@/lib/model-lifecycle-events";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useIsMobileShell } from "@/hooks/use-mobile";

import type { AudioGalleryClip } from "./api";
import {
  type AudioBusy,
  type AudioGenerationPhase,
  audioGenerationPresentation,
} from "./audio-page-policy";
import { type CreateMode, deviceSizeBytes } from "./audio-workspace-utils";
import { audioCapabilityLine, audioModelsForTask } from "./catalog";
import { galleryCache, useAudioGallery, useWorkflowHistory } from "./hooks/use-audio-gallery";
import { useAudioHandoff } from "./hooks/use-audio-handoff";
import { useAudioModelSlot } from "./hooks/use-audio-model-slot";
import { useCloneGeneration } from "./hooks/use-clone-generation";
import { useConvertGeneration } from "./hooks/use-convert-generation";
import { useSpeechGeneration } from "./hooks/use-speech-generation";
import { useSttSidecar } from "./hooks/use-stt-sidecar";
import { useTranscription } from "./hooks/use-transcription";
import {
  CloneFooter,
  CloneOutput,
  CloneRail,
  clonePageModels,
} from "./pages/clone-page";
import {
  ConvertFooter,
  ConvertOutput,
  ConvertRail,
  convertPageModels,
} from "./pages/convert-page";
import { MusicOutput, MusicRail, musicPageModels } from "./pages/music-page";
import { SpeakOutput, SpeakRail, speakPageModels } from "./pages/speak-page";
import { type GenerateBlocker, TtsFooter } from "./pages/tts-workspace";
import { TranscribeOutput, TranscribeRail } from "./pages/transcribe-page";
import { type AudioPickerRow, audioRowMatchesWorkflow } from "./picker-filter";
import { useAudioCloneStore } from "./stores/audio-clone-store";
import { useAudioConvertStore } from "./stores/audio-convert-store";
import { useAudioWorkspaceStore } from "./stores/audio-workspace-store";
import { AudioToolPanels } from "./tools/tool-panel-host";
import {
  AUDIO_WORKFLOWS,
  type AudioWorkflowId,
  audioWorkflowTab,
  clipWorkflow,
  isAudioWorkflowId,
  loadedModelRunsWorkflow,
  slotForWorkflow,
} from "./workflows";

const MODELS_BY_MODE: Record<CreateMode, ModelOption[]> = {
  speak: audioModelsForTask("tts"),
  transcribe: audioModelsForTask("stt"),
};

// Music GGUFs carry text-to-audio. The Hub rows it adds still pass the speech runtime gate.
const HUB_TASKS_BY_MODE = {
  speak: ["text-to-speech", "text-to-audio"],
  transcribe: ["automatic-speech-recognition"],
} as const;

function reuseConvertInputs(clip: AudioGalleryClip) {
  const store = useAudioConvertStore.getState();
  const sourceId = clip.source_clip_id ?? clip.source_input_id ?? null;
  if (sourceId) {
    store.setSource({
      kind: clip.source_clip_id ? "clip" : "input",
      id: sourceId,
      name: clip.source_name ?? "Recording",
      durationS: null,
    });
  }
  const target = clip.voice_id
    ? { kind: "voice" as const, id: clip.voice_id }
    : clip.target_clip_id
      ? { kind: "clip" as const, id: clip.target_clip_id }
      : clip.target_input_id
        ? { kind: "input" as const, id: clip.target_input_id }
        : null;
  if (target) {
    store.setTarget({
      ...target,
      name: clip.reference_name ?? "Target voice",
      durationS: null,
    });
  } else if (clip.target_builtin) {
    store.setBuiltinVoice(clip.target_builtin);
  }
}

export function AudioPage({
  active = true,
  onInitialReady,
}: {
  active?: boolean;
  onInitialReady?: () => void;
}) {
  const initialReadySent = useRef(false);
  // Clear the floating sidebar toggle on mobile.
  const isMobileShell = useIsMobileShell();
  const { pinned } = useSidebar();
  const [mode, setMode] = useState<CreateMode>("speak");
  const workflow = useAudioWorkspaceStore((state) => state.workflow);
  // The page shown follows the slot: a release that fails puts `mode` back on Transcribe before the
  // workflow catches up.
  const pageWorkflow: AudioWorkflowId =
    slotForWorkflow(workflow) === mode
      ? workflow
      : mode === "transcribe"
        ? "transcribe"
        : "speak";
  const ttsWorkflow = pageWorkflow === "transcribe" ? "speak" : pageWorkflow;
  const workflowTab = audioWorkflowTab(pageWorkflow);
  const { rootStyle: railRootStyle } = useMediaRailWidth("audio");
  const tourSteps = useMemo(
    () => buildAudioTourSteps({ workflow: pageWorkflow }),
    [pageWorkflow],
  );
  const tour = useGuidedTourController({
    id: "audio",
    steps: tourSteps,
    enabled: active,
  });
  const [selectorOpen, setSelectorOpen] = useState(false);
  const [busy, setBusy] = useState<AudioBusy>(null);
  const busyRef = useRef<AudioBusy>(busy);
  busyRef.current = busy;
  const [generationPhase, setGenerationPhase] =
    useState<AudioGenerationPhase>(null);
  const generationPhaseRef = useRef<AudioGenerationPhase>(generationPhase);
  const updateGenerationPhase = useCallback(
    (nextPhase: AudioGenerationPhase) => {
      generationPhaseRef.current = nextPhase;
      setGenerationPhase(nextPhase);
    },
    [],
  );

  const [status, setStatus] = useState<InferenceStatusResponse | null>(null);
  const generateAbort = useRef<AbortController | null>(null);
  const handleStopGeneration = useCallback(() => {
    const controller = generateAbort.current;
    if (!controller || controller.signal.aborted) return;
    updateGenerationPhase("stopping");
    controller.abort();
  }, [updateGenerationPhase]);
  const ttsStatusRefreshGeneration = useRef(0);
  const activeRef = useRef(active);
  activeRef.current = active;
  const modeRef = useRef(mode);
  modeRef.current = mode;

  const {
    attach: attachSettingsScroll,
    onScroll: onSettingsScroll,
    className: settingsFadeClass,
  } = useScrollFades();
  const [advancedOpen, setAdvancedOpen] = usePersistedToggle(
    "unsloth_audio_advanced_open",
  );
  // Read at load time; the handler below ejects so a change takes effect.
  const [audioDevice, setAudioDeviceState] = usePersistedChoice(
    "unsloth_audio_device",
    "auto",
  );

  const refreshStatus = useCallback(async () => {
    const generation = ++ttsStatusRefreshGeneration.current;
    try {
      const next = await getInferenceStatus();
      if (generation !== ttsStatusRefreshGeneration.current) return;
      setStatus(next);
    } catch {
      if (generation !== ttsStatusRefreshGeneration.current) return;
      // Do not leave Generate enabled against residency the backend can no longer confirm. A later
      // refresh adopts the recovered runtime.
      setStatus(null);
    }
  }, []);

  const isMac = usePlatformStore((s) => s.deviceType) === "mac";

  const {
    lastSttRepo,
    selectedSttRepo,
    setSelectedSttRepo,
    sttLoadedModel,
    sttLoadedEngine,
    downloadedSttArtifacts,
    selectedSttRepoRef,
    sttStatusRefreshGeneration,
    audioCppRuntime,
    sttLoadGeneration,
    sttGgufVariants,
    setLastSttRepo,
    sttLoadingGeneration,
    sttLoadAbort,
    deferredSttLoad,
    refreshSttStatus,
    sttSelected,
    sttReady,
    releaseTranscribeSelection,
    ensureSttLoaded,
  } = useSttSidecar({ setBusy, activeRef });

  const {
    transcript,
    setTranscript,
    transcribedName,
    setTranscribedName,
    transcriptModel,
    setTranscriptModel,
    transcriptRecord,
    setTranscriptRecord,
    transcriptExported,
    setTranscriptExported,
    transcriptVersion,
    transcriptionStartedAt,
    setTranscriptionStartedAt,
    transcriptionFinishedAt,
    transcriptionStopping,
    setTranscriptionStopping,
    transcriptionProgress,
    transcriptError,
    setTranscriptError,
    isRecording,
    micRequestPending,
    recordingSupported,
    transcriptionAbort,
    stopAndDiscardRecording,
    clearTranscript,
    confirmTranscriptReplacement,
    handleRecordToggle,
    handleTranscribeFile,
    handleCopyTranscript,
    handleDownloadTranscript,
  } = useTranscription({
    active,
    activeRef,
    busyRef,
    setBusy,
    lastSttRepo,
    selectedSttRepoRef,
    setSelectedSttRepo,
    sttLoadedModel,
    sttLoadedEngine,
    ensureSttLoaded,
    setLastSttRepo,
    refreshSttStatus,
  });

  const {
    fallbackClip,
    setFallbackClip,
    loadingMoreRef,
    clips,
    hasMore,
    selectedId,
    setSelectedId,
    srcById,
    ensureClipSrc,
    refreshGallery,
    loadMore,
    selectClip,
    handleDeleteClip,
    handleArchiveClip,
    handleTogglePin,
    historyReorder,
    handleClearGallery,
    handleDownloadClip,
    handleDownloadFallbackClip,
    handleDownloadClipById,
    handleCopyPrompt,
  } = useAudioGallery({ active });

  // Convert reads the running server task to say whether its next run reloads. A Clone run, even a
  // stopped one, can restart audio.cpp after its request returns, so read it again on arriving.
  useEffect(() => {
    if (active && ttsWorkflow === "convert" && initialReadySent.current) {
      void refreshStatus();
    }
  }, [active, ttsWorkflow, refreshStatus]);

  // Resync on activation: another tab may have loaded/unloaded models meanwhile.
  useEffect(() => {
    if (!active) return;
    if (initialReadySent.current) {
      void refreshStatus();
      void refreshSttStatus();
      void refreshGallery(undefined, galleryCache.clips.length);
      return;
    }
    let cancelled = false;
    void (async () => {
      const [initialClips] = await Promise.all([
        refreshGallery(),
        refreshStatus(),
        refreshSttStatus(),
      ]);
      const initialSelection =
        initialClips.find((clip) => clip.id === galleryCache.selectedId) ??
        initialClips[0];
      if (initialSelection) await ensureClipSrc(initialSelection);
      if (cancelled || initialReadySent.current) return;
      initialReadySent.current = true;
      onInitialReady?.();
    })();
    return () => {
      cancelled = true;
    };
  }, [
    active,
    ensureClipSrc,
    onInitialReady,
    refreshGallery,
    refreshStatus,
    refreshSttStatus,
  ]);

  // Activation alone is not enough: the loaded-models indicator can eject the TTS model out from
  // under a page that stays active, leaving Generate enabled against an empty slot.
  useEffect(() => {
    if (!active) return;
    return subscribeModelLifecycle(({ runtime, loading }) => {
      if (loading) return;
      if (runtime === "chat") void refreshStatus();
      if (runtime === "stt") void refreshSttStatus();
    });
  }, [active, refreshStatus, refreshSttStatus]);

  const {
    replayQueuedTtsPick,
    transitionMode,
    transitionWorkflow,
    pendingTranscribeRelease,
    handleModelSelect,
    pickRecommendedModel,
    handleEject,
    trainedTtsModels,
  } = useAudioModelSlot({
    active,
    activeRef,
    busy,
    busyRef,
    setBusy,
    mode,
    setMode,
    modeRef,
    generationPhaseRef,
    handleStopGeneration,
    refreshStatus,
    audioDevice,
    isMac,
    status,
    isRecording,
    stopAndDiscardRecording,
    releaseTranscribeSelection,
    ensureSttLoaded,
    sttGgufVariants,
    selectedSttRepoRef,
    setSelectedSttRepo,
    deferredSttLoad,
    sttLoadingGeneration,
    sttLoadGeneration,
    sttLoadAbort,
    audioCppRuntime,
    selectedSttRepo,
    sttReady,
    sttStatusRefreshGeneration,
  });

  // The clip a run just made: its player takes focus once and screen readers hear it is ready.
  const [freshClipId, setFreshClipId] = useState<string | null>(null);
  const [announcement, setAnnouncement] = useState("");
  const selectGeneratedClip = useCallback(
    (id: string, keepFallback?: boolean) => {
      selectClip(id, keepFallback);
      setFreshClipId(id);
      setAnnouncement("New clip ready.");
    },
    [selectClip],
  );
  const clearFreshClip = useCallback(() => setFreshClipId(null), []);
  useEffect(() => {
    if (busy === "generating") setAnnouncement("");
  }, [busy]);

  const {
    prompt,
    setPrompt,
    audioInstructions,
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
    handleGenerate,
    generationError,
    setGenerationError,
    toolContext,
    toolValues,
    handleToolValueChange,
    toolBlocker,
    claimedOptions,
  } = useSpeechGeneration({
    // Clone keeps its own drafts and runs through its own hook; this one serves Speak and Music.
    workflow: ttsWorkflow === "music" ? "music" : "speak",
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
    selectClip: selectGeneratedClip,
    setFallbackClip,
    setSelectedId,
    pendingTranscribeRelease,
    replayQueuedTtsPick,
  });

  // The clip's Transcribe button uses Transcribe's model, else one already on disk, so it works
  // without a trip to Settings.
  const lastSttDownloaded =
    lastSttRepo !== null &&
    downloadedSttArtifacts.some(
      (artifact) => artifact.repoId.toLowerCase() === lastSttRepo.toLowerCase(),
    );
  const cloneSttRepo =
    selectedSttRepo ??
    (lastSttDownloaded ? lastSttRepo : null) ??
    downloadedSttArtifacts[0]?.repoId ??
    lastSttRepo ??
    null;
  const clone = useCloneGeneration({
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
    selectClip: selectGeneratedClip,
    setFallbackClip,
    setSelectedId,
    pendingTranscribeRelease,
    replayQueuedTtsPick,
    audioOptionSpecs,
    audioOptionValues,
    sttRepo: cloneSttRepo,
  });
  const convert = useConvertGeneration({
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
    selectClip: selectGeneratedClip,
    setFallbackClip,
    setSelectedId,
    pendingTranscribeRelease,
    replayQueuedTtsPick,
    audioOptionSpecs,
    audioOptionValues,
    sttRepo: cloneSttRepo,
  });
  const generationPresentation = audioGenerationPresentation(
    generationPhase,
    convert.runningNotice ?? undefined,
  );

  const { navigateSelf } = useAudioHandoff({
    active,
    busy,
    busyRef,
    mode,
    modeRef,
    handleModelSelect,
    transitionMode,
    transitionWorkflow,
    refreshGallery,
    loadMore,
    loadingMoreRef,
    selectClip,
  });

  // The sidebar and the More flyout ask for a workflow; the page takes it through the same gate as
  // its own tabs, and a refused switch is not retried.
  const requestedWorkflow = useAudioWorkspaceStore(
    (state) => state.requestedWorkflow,
  );
  useEffect(() => {
    if (!active || requestedWorkflow === null) return;
    useAudioWorkspaceStore.getState().clearRequestedWorkflow();
    transitionWorkflow(requestedWorkflow);
  }, [active, requestedWorkflow, transitionWorkflow]);

  // A failed Transcribe release puts `mode` back on Transcribe from inside the slot; the workflow follows.
  const syncedMode = useRef(mode);
  useEffect(() => {
    if (syncedMode.current === mode) return;
    syncedMode.current = mode;
    const store = useAudioWorkspaceStore.getState();
    if (slotForWorkflow(store.workflow) === mode) return;
    store.commitWorkflow(
      mode === "transcribe" ? "transcribe" : musicGeneration ? "music" : "speak",
    );
  }, [mode, musicGeneration]);

  // Opening Audio with a music model already loaded lands on Music, as the one page used to adapt to it.
  const adoptedLoadedModel = useRef(false);
  useEffect(() => {
    if (adoptedLoadedModel.current || status === null) return;
    adoptedLoadedModel.current = true;
    if (modeRef.current === "speak" && ttsLoaded && musicGeneration) {
      useAudioWorkspaceStore.getState().adoptWorkflow("music");
    }
  }, [status, ttsLoaded, musicGeneration]);

  const {
    visibleClips,
    selectedClip,
    selectedClipSrc,
  } = useWorkflowHistory({
    workflow: ttsWorkflow,
    enabled: active && mode === "speak",
    clips,
    hasMore,
    selectedId,
    srcById,
    fallbackClip,
    loadMore,
    loadingMoreRef,
    selectClip,
  });
  const handleClearWorkflowGallery = useCallback(
    () => handleClearGallery(ttsWorkflow),
    [handleClearGallery, ttsWorkflow],
  );
  const handleUseTextAgain = useCallback(
    (clip: AudioGalleryClip) => {
      const target = clipWorkflow(clip);
      if (!transitionWorkflow(target)) return;
      if (target === "clone") useAudioCloneStore.getState().setText(clip.prompt);
      else if (target === "convert") reuseConvertInputs(clip);
      else setPrompt(clip.prompt);
    },
    [transitionWorkflow, setPrompt],
  );
  const handleSendToConvert = useCallback(
    (clip: AudioGalleryClip) => {
      if (!transitionWorkflow("convert")) return;
      useAudioConvertStore.getState().setSource({
        kind: "clip",
        id: clip.id,
        name: clip.prompt,
        durationS: clip.duration_s ?? null,
      });
    },
    [transitionWorkflow],
  );

  // What the rail shows follows the page; generation still runs on what is actually loaded.
  const pageModelLoaded =
    ttsLoaded &&
    loadedModelRunsWorkflow({
      workflow: ttsWorkflow,
      audioWorkflows: status?.audio_workflows,
      music: musicGeneration,
    });
  // The model this page last loaded, shown in the picker as not loaded while another model holds the slot.
  const lastPageModel = useAudioWorkspaceStore(
    (state) => state.lastModelByWorkflow[pageWorkflow] ?? null,
  );
  const openSelector = useCallback(() => setSelectorOpen(true), []);
  const chooseModelAction = { label: "Choose a model", onClick: openSelector };
  const handlePickRecommended = useCallback(
    (id: string) => void pickRecommendedModel(id),
    [pickRecommendedModel],
  );
  const recommendedCloneActions = clonePageModels(MODELS_BY_MODE.speak, isMac)
    .slice(0, 2)
    .map((model) => ({
      label: `use ${model.name}`,
      onClick: () => handlePickRecommended(model.id),
    }));
  const recommendedConvertActions = convertPageModels(
    MODELS_BY_MODE.speak,
    isMac,
  )
    .slice(0, 2)
    .map((model) => ({
      label: `use ${model.name}`,
      onClick: () => handlePickRecommended(model.id),
    }));
  // RVC, Seed-VC and MeanVC2 only convert; Speak must not send their users to Clone.
  const loadedConvertsOnly =
    !!status?.audio_workflows?.includes("convert") &&
    !status.audio_workflows.includes("clone");
  // Why Generate is off, in words under the button, with the fix as an action where there is one.
  const generateBlocker: GenerateBlocker | null =
    busy === "loading"
      ? { reason: "Waiting for the model to finish loading." }
      : busy !== null
        ? null
        : !ttsLoaded
          ? {
              reason:
                ttsWorkflow === "music"
                  ? "Load a music model to generate."
                  : ttsWorkflow === "clone"
                    ? "Load a model that can clone a voice."
                    : ttsWorkflow === "convert"
                      ? "Load a model that can convert a voice."
                      : "Load a speech model to generate.",
              actions:
                ttsWorkflow === "clone"
                  ? [chooseModelAction, ...recommendedCloneActions]
                  : ttsWorkflow === "convert"
                    ? [chooseModelAction, ...recommendedConvertActions]
                    : [chooseModelAction],
            }
          : !pageModelLoaded && ttsWorkflow === "clone"
            ? {
                reason: "The loaded model cannot clone a voice.",
                actions: [
                  { label: "Choose a model that can clone", onClick: openSelector },
                  ...recommendedCloneActions,
                  {
                    label: "open Speak",
                    onClick: () => transitionWorkflow("speak"),
                  },
                ],
              }
            : !pageModelLoaded && ttsWorkflow === "convert"
              ? {
                  reason: "The loaded model cannot convert a voice.",
                  actions: [
                    {
                      label: "Choose a model that can convert",
                      onClick: openSelector,
                    },
                    ...recommendedConvertActions,
                  ],
                }
            : !pageModelLoaded
            ? musicGeneration
              ? {
                  reason: "The loaded model makes music.",
                  actions: [
                    { label: "Choose a speech model", onClick: openSelector },
                    {
                      label: "open Music",
                      onClick: () => transitionWorkflow("music"),
                    },
                  ],
                }
              : ttsWorkflow === "speak" && loadedConvertsOnly
                ? {
                    reason: "The loaded model converts recordings.",
                    actions: [
                      { label: "Choose a speech model", onClick: openSelector },
                      {
                        label: "open Convert",
                        onClick: () => transitionWorkflow("convert"),
                      },
                    ],
                  }
              : ttsWorkflow === "speak"
                ? {
                    reason: "The loaded model needs a voice to clone.",
                    actions: [
                      { label: "Choose a speech model", onClick: openSelector },
                      {
                        label: "open Clone",
                        onClick: () => transitionWorkflow("clone"),
                      },
                    ],
                  }
                : {
                    reason: "The loaded model makes speech.",
                    actions: [
                      { label: "Choose a music model", onClick: openSelector },
                      {
                        label: "open Speak",
                        onClick: () => transitionWorkflow("speak"),
                      },
                    ],
                  }
            : ttsWorkflow === "clone"
              ? clone.blocker
              : ttsWorkflow === "convert"
                ? convert.blocker
              : !prompt.trim() && !lyricsOptional
              ? {
                  reason:
                    ttsWorkflow === "music"
                      ? "Write the lyrics to generate."
                      : "Type the text to speak.",
                }
              : musicNeedsDescription && !audioInstructions.trim()
                ? {
                    reason:
                      "Add a music description. This model needs one beside the lyrics.",
                  }
                : toolBlocker
                  ? { reason: toolBlocker }
                  : null;
  // A failed run's reason stays under Generate with a way out, until the next run or a new model.
  const pageGenerationError =
    ttsWorkflow === "clone"
      ? clone.generationError
      : ttsWorkflow === "convert"
        ? convert.generationError
        : generationError;
  const generateFailure: GenerateBlocker | null = pageGenerationError
    ? {
        // Runtime reasons often end without a stop, which would run into the action.
        reason: /[.!?]$/.test(pageGenerationError)
          ? pageGenerationError
          : `${pageGenerationError}.`,
        actions: [{ label: "Choose a different model", onClick: openSelector }],
      }
    : null;
  const setCloneGenerationError = clone.setGenerationError;
  const setConvertGenerationError = convert.setGenerationError;
  useEffect(() => {
    setGenerationError(null);
    setCloneGenerationError(null);
    setConvertGenerationError(null);
  }, [
    status?.active_model,
    setGenerationError,
    setCloneGenerationError,
    setConvertGenerationError,
  ]);
  useEffect(() => {
    if (busy === "generating") setGenerationError(null);
    if (busy === "generating") setCloneGenerationError(null);
    if (busy === "generating") setConvertGenerationError(null);
  }, [busy, setGenerationError, setCloneGenerationError, setConvertGenerationError]);
  // How long the current run has taken, shown beside its phase.
  const [elapsedSeconds, setElapsedSeconds] = useState<number | null>(null);
  useEffect(() => {
    if (busy !== "generating") {
      setElapsedSeconds(null);
      return;
    }
    const startedAt = Date.now();
    setElapsedSeconds(0);
    const timer = window.setInterval(
      () => setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000)),
      1000,
    );
    return () => window.clearInterval(timer);
  }, [busy]);
  const canGenerate =
    mode === "speak" &&
    busy === null &&
    !generationPresentation &&
    pageModelLoaded &&
    generateBlocker === null;
  const shortcutLabel = isMac ? "⌘ Enter" : "Ctrl+Enter";
  // Mod+Enter generates from anywhere on the page, as the button would.
  const pageRootRef = useRef<HTMLDivElement | null>(null);
  const generateShortcut = useRef<() => void>(() => {});
  const handlePageGenerate =
    ttsWorkflow === "clone"
      ? clone.handleGenerate
      : ttsWorkflow === "convert"
        ? convert.handleGenerate
        : handleGenerate;
  generateShortcut.current = () => {
    if (canGenerate) void handlePageGenerate();
  };
  useEffect(() => {
    if (!active) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        event.key !== "Enter" ||
        !(event.metaKey || event.ctrlKey) ||
        event.isComposing ||
        !(event.target instanceof Node) ||
        !pageRootRef.current?.contains(event.target)
      )
        return;
      event.preventDefault();
      generateShortcut.current();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [active]);
  const selectorModels =
    mode === "speak"
      ? ttsWorkflow === "music"
        ? musicPageModels(MODELS_BY_MODE.speak, isMac)
        : ttsWorkflow === "clone"
          ? clonePageModels(MODELS_BY_MODE.speak, isMac)
          : ttsWorkflow === "convert"
            ? convertPageModels(MODELS_BY_MODE.speak, isMac)
          : speakPageModels(MODELS_BY_MODE.speak, isMac)
      : MODELS_BY_MODE[mode];
  // Downloaded rows the picker adds from the cache are narrowed to this page's task the same way.
  const selectorRowFilter = useCallback(
    (row: AudioPickerRow) =>
      audioRowMatchesWorkflow(row, pageWorkflow),
    [pageWorkflow],
  );
  const sttOnDeviceModels = downloadedSttArtifacts.map((artifact) => {
    const catalogModel = MODELS_BY_MODE.transcribe.find(
      (model) => model.id.toLowerCase() === artifact.repoId.toLowerCase(),
    );
    const size = sttModelSize(artifact.sidecarKey);
    return {
      ...(catalogModel ?? {
        id: artifact.repoId,
        name: artifact.repoId.split("/").pop() || artifact.repoId,
        description: "Speech-to-text",
      }),
      isGguf: artifact.engine !== "transformers",
      deviceQuant:
        artifact.engine === "mtmd"
          ? "Q8_0"
          : artifact.engine === "gguf"
            ? "F16"
            : undefined,
      deviceSize: size || undefined,
      deviceSizeBytes: size ? deviceSizeBytes(size) : undefined,
      deviceLoaded:
        artifact.sidecarKey === sttLoadedModel &&
        artifact.engine === sttLoadedEngine,
    } satisfies ModelOption;
  });
  // Arriving on a page never loads anything: when the resident model cannot run it, the picker
  // shows the page's last model, unloaded, so one pick brings it back.
  const showLastPageModel =
    mode === "speak" && !pageModelLoaded && lastPageModel !== null;
  const selectorValue =
    mode === "speak"
      ? showLastPageModel
        ? lastPageModel
        : (status?.active_model ?? undefined)
      : (selectedSttRepo ?? undefined);

  const capabilityLine =
    mode === "speak"
      ? showLastPageModel
        ? `${lastPageModel.split("/").pop()} is not loaded. Pick it above to load it.`
        : ttsLoaded
        ? pageModelLoaded
          ? audioCapabilityLine(
              musicGeneration
                ? "music"
                : ttsWorkflow === "clone" || ttsWorkflow === "convert"
                  ? ttsWorkflow
                  : "tts",
              status?.audio_type,
            )
          : ttsWorkflow === "clone"
            ? "The loaded model cannot clone a voice."
            : ttsWorkflow === "convert"
              ? "The loaded model cannot convert a voice."
            : musicGeneration
            ? "The loaded model makes music. Pick a speech model."
            : ttsWorkflow === "speak" && loadedConvertsOnly
              ? "The loaded model converts recordings. Pick a speech model."
            : ttsWorkflow === "speak"
              ? "The loaded model needs a voice to clone. Pick a speech model."
              : "The loaded model is not a music model."
        : status?.active_model
          ? "The loaded model is not a TTS audio model."
          : ttsWorkflow === "music"
            ? "No music model loaded."
            : ttsWorkflow === "clone"
              ? "No voice cloning model loaded."
              : ttsWorkflow === "convert"
                ? "No voice conversion model loaded."
              : "No TTS model loaded."
      : sttSelected
        ? audioCapabilityLine("stt", sttReady ? "ready" : "loading")
        : "No transcription model selected.";

  return (
    <div
      {...{ [MEDIA_RAIL_ROOT_ATTR]: "" }}
      style={railRootStyle}
      ref={pageRootRef}
      className="@container relative flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden pt-[var(--studio-content-top-inset,0px)]"
    >
      {/* Page-level, so the handle covers the divider through the header too. */}
      <MediaRailResizeHandle kind="audio" placement="page" className="hidden @[50rem]:block" />
      {/* Portals to body, and this page stays mounted off-route, so gate it like the composer. */}
      {active && <GuidedTour {...tour.tourProps} />}
      {/* Keep the tabs centered over the preview at every width. The model rail holds at its
          (draggable) width when space permits and shrinks only to preserve the controls. */}
      <div className="pointer-events-none relative z-40 grid h-[calc(48px*var(--ui-space-scale,1))] shrink-0 grid-cols-[minmax(0,var(--media-rail-width,calc(408px*var(--ui-space-scale,1))))_minmax(13rem,1fr)] @max-[30rem]:grid-cols-[minmax(0,1fr)_auto]">
        <div
          className={cn(
            "pointer-events-none flex h-full min-w-0 items-start overflow-hidden @[50rem]:border-r @[50rem]:border-border/60",
            isMobileShell
            ? "pl-12"
            : // Collapsed desktop sidebar: clear the titlebar buttons, as Chat does.
              !pinned && isTauri
              ? "pl-[var(--studio-collapsed-chat-controls-inset,0.75rem)]"
              : "pl-[var(--studio-media-header-left-inset,1.5rem)]",
          )}
        >
          {/* A long resident model name must yield to the mode pill instead of painting over it. */}
          <div className="pointer-events-auto flex min-w-0 max-w-full items-center gap-2 overflow-hidden pt-[var(--studio-chat-header-padding-top,11px)]">
            <ModelSelector
              triggerDataTour="audio-model"
              models={selectorModels}
              additionalOnDeviceModels={
                mode === "transcribe"
                  ? sttOnDeviceModels
                  : ttsWorkflow === "speak"
                    ? trainedTtsModels
                    : []
              }
              rowFilter={selectorRowFilter}
              loadedModelIdOverride={
                mode === "transcribe" && sttReady
                  ? (selectedSttRepo ?? undefined)
                  : undefined
              }
              loaded={
                mode === "transcribe"
                  ? sttReady
                  : showLastPageModel
                    ? false
                    : undefined
              }
              value={selectorValue}
              onValueChange={handleModelSelect}
              onEject={
                busy === null && selectorValue && !showLastPageModel
                  ? handleEject
                  : undefined
              }
              variant="ghost"
              className="!h-[calc(34px*var(--ui-space-scale,1))] max-w-full gap-1 overflow-hidden pl-3 pr-1 @[68rem]:gap-2 @[68rem]:pl-4 @[68rem]:pr-2"
              triggerLabelClassName="text-ui-14 @[68rem]:text-ui-16"
              task={HUB_TASKS_BY_MODE[mode]}
              catalog={AUDIO_CATALOG}
              // TTS/ASR come from the checkpoint's own tokenizer, not a curated recipe, so any publisher's
              // audio repo loads here.
              communityModelPolicy="search-only"
              hubCapability="audio"
              placeholder="Select audio model"
              open={active && selectorOpen}
              onOpenChange={(o) => setSelectorOpen(active && o)}
            />
          </div>
        </div>
        <div className="grid h-full min-w-0 grid-cols-[1fr_auto_auto] gap-2 @[50rem]:grid-cols-[1fr_auto_1fr] @[50rem]:gap-0">
          <div className="pointer-events-auto col-start-2 justify-self-end pt-[var(--studio-chat-header-padding-top,11px)] @[50rem]:justify-self-center">
            {workflowTab.createTrain ? (
              <PillTabs
                ariaLabel="Page mode"
                // Always "create": Train navigates away, so the pill never latches.
                value="create"
                onValueChange={(v) => {
                  if (v !== "train") return;
                  toast.info(
                    "Audio fine-tuning lives on the Train page. Unsloth trains TTS and STT models there. Pick an audio model and appropriate dataset.",
                    { duration: 8000 },
                  );
                  void navigateSelf({ to: "/studio" });
                }}
                fit={true}
                className="h-[calc(34px*var(--ui-space-scale,1))] [&>button]:h-[calc(34px*var(--ui-space-scale,1))] [&>button]:px-3 @[68rem]:[&>button]:px-11 @max-[30rem]:[&>button]:px-2.5 @max-[30rem]:[&>button>span]:sr-only"
                tabs={[
                  {
                    value: "create",
                    label: "Create",
                    icon: (
                      <HugeiconsIcon icon={SparklesIcon} className="size-3.5" />
                    ),
                  },
                  {
                    value: "train",
                    label: "Train",
                    icon: (
                      <HugeiconsIcon
                        icon={TestTubeOutlineIcon}
                        className="size-3.5"
                      />
                    ),
                  },
                ]}
              />
            ) : null}
          </div>
          <div className="pointer-events-none col-start-3 flex min-w-0 items-start justify-end pr-2 pt-[var(--studio-chat-header-padding-top,11px)]">
            <div className="pointer-events-auto flex min-w-0 items-center gap-2">
              <LibraryPageLink
                tab="audio"
                labelClassName="hidden @[50rem]:inline"
                arrowClassName="hidden @[50rem]:block"
              />
            </div>
          </div>
        </div>
      </div>
      {/* Below 50rem the panes stack and the page scrolls as one column, matching Images and Video:
          side by side, the rail plus a usable preview needs more width. */}
      <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-y-auto overflow-x-hidden @[50rem]:flex-row @[50rem]:overflow-hidden">
        <div
          data-tour="audio-settings"
          className="flex w-full shrink-0 flex-col border-b border-border/60 @[50rem]:w-[min(var(--media-rail-width,calc(408px*var(--ui-space-scale,1))),calc(100%-13rem))] @[50rem]:overflow-hidden @[50rem]:border-r @[50rem]:border-b-0"
        >
          <div
            ref={attachSettingsScroll}
            onScroll={onSettingsScroll}
            className={cn(
              "hover-scrollbar flex min-h-0 flex-1 flex-col gap-4 px-10 max-sm:px-5 pt-9 pb-6 @[50rem]:overflow-y-auto",
              mode === "speak"
                ? "panel-scroll-fade-action"
                : "panel-scroll-fade",
              settingsFadeClass,
            )}
          >
            {/* Same heading treatment as the Images and Video Create panes, so the media panes stay level (#7986). */}
            <div className="mb-2 grid gap-1.5">
              <h2 className="flex items-center gap-2 font-heading text-xl font-medium leading-none text-foreground">
                <HugeiconsIcon
                  icon={workflowTab.icon}
                  className="size-[calc(18px*var(--ui-space-scale,1))] shrink-0"
                />
                {workflowTab.heading}
              </h2>
              <p className="text-ui-11p5 leading-snug text-muted-foreground">
                {workflowTab.hint}
              </p>
              {/* The always-on capability line: which task the selected model actually does. */}
              <p className="text-xs leading-snug text-muted-foreground">
                {capabilityLine}
              </p>
            </div>

            {/* Five tabs can outgrow a narrow rail, so this wrapper scrolls them. Overflow on the tab list
                itself clipped the selected pill's edge and shadow; the padding leaves room for both. */}
            <div className="-m-1.5 shrink-0 self-start overflow-x-auto p-1.5 [max-width:calc(100%+0.75rem)] [scrollbar-width:none]">
              <PillTabs
                dataTour="audio-mode"
                ariaLabel="Audio workflow"
                value={pageWorkflow}
                onValueChange={(v) => {
                  if (isAudioWorkflowId(v)) transitionWorkflow(v);
                }}
                fit={true}
                className="h-[calc(30px*var(--ui-space-scale,1))] max-w-none [&>button]:h-[calc(30px*var(--ui-space-scale,1))] [&>button]:shrink-0 [&>button]:px-2.5"
                tabs={AUDIO_WORKFLOWS.map(({ id, label }) => ({
                  value: id,
                  label,
                }))}
              />
            </div>

            {mode === "speak" ? (
              (() => {
                const railProps = {
                  prompt,
                  setPrompt,
                  toolPanels: (
                    <AudioToolPanels
                      workflow={ttsWorkflow === "music" ? "music" : "speak"}
                      ctx={toolContext}
                      values={toolValues}
                      onChange={handleToolValueChange}
                      specs={audioOptionSpecs}
                      disabled={busy === "generating"}
                      core={{ text: prompt }}
                    />
                  ),
                  audioDevice,
                  busy,
                  isRecording,
                  status,
                  setAudioDeviceState,
                  ttsLoaded,
                  handleEject,
                  samplingControls,
                  audioOptionSpecs,
                  claimedOptions,
                  advancedOpen,
                  setAdvancedOpen,
                  temperature,
                  mossFrameLimit,
                  handleTemperatureChange,
                  musicSeconds,
                  musicRange,
                  setMinimaxMaxSeconds,
                  cudaMusicGeneration,
                  mossMaxSeconds,
                  mossMaxSecondsLimit,
                  setMossMaxSeconds,
                  maxTokens,
                  setMaxTokens,
                  audioOptionValues,
                  handleAudioOptionChange,
                  handleAudioOptionsReset,
                };
                return ttsWorkflow === "music" ? (
                  <MusicRail {...railProps} />
                ) : ttsWorkflow === "clone" ? (
                  <CloneRail {...railProps} clone={clone} historyClips={clips} />
                ) : ttsWorkflow === "convert" ? (
                  <ConvertRail
                    {...railProps}
                    convert={convert}
                    historyClips={clips}
                  />
                ) : (
                  <SpeakRail {...railProps} />
                );
              })()
            ) : (
              <TranscribeRail
                recordingSupported={recordingSupported}
                isRecording={isRecording}
                sttSelected={sttSelected}
                lastSttRepo={lastSttRepo}
                busy={busy}
                micRequestPending={micRequestPending}
                handleRecordToggle={handleRecordToggle}
                handleTranscribeFile={handleTranscribeFile}
              />
            )}
          </div>
          {mode === "speak" ? (
            /* The scroll mask provides the fade; leave the footer unpainted to avoid dark-mode banding. */
            <div className="relative z-10 flex shrink-0 justify-center px-10 pt-0.5 pb-4">
              {ttsWorkflow === "clone" ? (
                <CloneFooter
                  busy={busy}
                  generationPresentation={generationPresentation}
                  handleStopGeneration={handleStopGeneration}
                  ttsLoaded={pageModelLoaded}
                  blocker={generateBlocker}
                  shortcutLabel={shortcutLabel}
                  error={generateFailure}
                  elapsedSeconds={elapsedSeconds}
                  clone={clone}
                />
              ) : ttsWorkflow === "convert" ? (
                <ConvertFooter
                  busy={busy}
                  generationPresentation={generationPresentation}
                  handleStopGeneration={handleStopGeneration}
                  ttsLoaded={pageModelLoaded}
                  blocker={generateBlocker}
                  shortcutLabel={shortcutLabel}
                  error={generateFailure}
                  elapsedSeconds={elapsedSeconds}
                  convert={convert}
                />
              ) : (
                <TtsFooter
                  busy={busy}
                  generationPresentation={generationPresentation}
                  handleStopGeneration={handleStopGeneration}
                  handleGenerate={handleGenerate}
                  ttsLoaded={pageModelLoaded}
                  prompt={prompt}
                  lyricsOptional={lyricsOptional}
                  musicNeedsDescription={musicNeedsDescription}
                  audioInstructions={audioInstructions}
                  blocker={generateBlocker}
                  shortcutLabel={shortcutLabel}
                  error={generateFailure}
                  elapsedSeconds={elapsedSeconds}
                />
              )}
            </div>
          ) : null}
        </div>

        <div
          data-tour="audio-output"
          className="relative flex min-h-[60dvh] min-w-0 flex-1 flex-col overflow-hidden @[50rem]:min-h-0"
        >
          {mode === "transcribe" ? (
            <div
              data-reload-snapshot-sensitive={
                transcript || transcribedName ? "" : undefined
              }
              className="flex min-h-0 flex-1 flex-col gap-3 p-6 px-10 @[50rem]:pt-[calc(60px*var(--ui-space-scale,1))]"
            >
              <TranscribeOutput
                transcriptionStartedAt={transcriptionStartedAt}
                transcriptionFinishedAt={transcriptionFinishedAt}
                transcriptionStopping={transcriptionStopping}
                transcriptionProgress={transcriptionProgress}
                setTranscriptionStopping={setTranscriptionStopping}
                transcriptionAbort={transcriptionAbort}
                transcript={transcript}
                handleCopyTranscript={handleCopyTranscript}
                handleDownloadTranscript={handleDownloadTranscript}
                transcribedName={transcribedName}
                transcriptModel={transcriptModel}
                transcriptRecord={transcriptRecord}
                transcriptExported={transcriptExported}
                transcriptError={transcriptError}
                busy={busy}
                active={active}
                mode={mode}
                confirmTranscriptReplacement={confirmTranscriptReplacement}
                transcriptVersion={transcriptVersion}
                setTranscript={setTranscript}
                setTranscribedName={setTranscribedName}
                setTranscriptModel={setTranscriptModel}
                setTranscriptRecord={setTranscriptRecord}
                setTranscriptError={setTranscriptError}
                setTranscriptExported={setTranscriptExported}
                setTranscriptionStartedAt={setTranscriptionStartedAt}
                clearTranscript={clearTranscript}
              />
            </div>
          ) : (
            <div className="flex min-h-0 flex-1 flex-col gap-4 p-6 px-10 @[50rem]:pt-[calc(60px*var(--ui-space-scale,1))]">
              {(() => {
                const outputProps = {
                  clips: visibleClips,
                  selectedClip,
                  selectedClipSrc,
                  srcById,
                  handleDownloadClip,
                  handleDeleteClip,
                  fallbackClip,
                  handleDownloadFallbackClip,
                  handleClearGallery: handleClearWorkflowGallery,
                  historyReorder,
                  hasMore,
                  loadMore,
                  selectedId,
                  selectClip,
                  handleTogglePin,
                  active,
                  handleArchiveClip,
                  handleDownloadClipById,
                  onUseTextAgain: handleUseTextAgain,
                  onSendToConvert: handleSendToConvert,
                  handleCopyPrompt,
                  freshClipId,
                  onFreshClipFocused: clearFreshClip,
                  announcement,
                };
                return ttsWorkflow === "music" ? (
                  <MusicOutput {...outputProps} modelReady={pageModelLoaded} />
                ) : ttsWorkflow === "clone" ? (
                  <CloneOutput
                    {...outputProps}
                    modelReady={pageModelLoaded}
                    recommendedModels={selectorModels}
                    onPickModel={handlePickRecommended}
                  />
                ) : ttsWorkflow === "convert" ? (
                  <ConvertOutput
                    {...outputProps}
                    useAgainLabel="Use these inputs again"
                    modelReady={pageModelLoaded}
                    recommendedModels={selectorModels}
                    onPickModel={handlePickRecommended}
                  />
                ) : (
                  <SpeakOutput {...outputProps} modelReady={pageModelLoaded} />
                );
              })()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
