// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stays mounted across tab switches (__root.tsx), so `active` gates polling, popovers and the recorder.

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

import { type AudioGalleryClip, fetchClipBlob } from "./api";
import {
  type AudioBusy,
  type AudioGenerationPhase,
  audioGenerationPresentation,
  instructionsFieldKind,
} from "./audio-page-policy";
import { type CreateMode, deviceSizeBytes } from "./audio-workspace-utils";
import { audioCapabilityLine, audioModelsForTask } from "./catalog";
import type { ClipSendHandlers } from "./components/clip-card";
import { WorkflowTitleMenu } from "./components/workflow-title-menu";
import { galleryCache, useAudioGallery, useWorkflowHistory } from "./hooks/use-audio-gallery";
import { useAudioHandoff } from "./hooks/use-audio-handoff";
import { useAudioModelSlot } from "./hooks/use-audio-model-slot";
import { useSpeechGeneration } from "./hooks/use-speech-generation";
import { useSttSidecar } from "./hooks/use-stt-sidecar";
import { useTranscription } from "./hooks/use-transcription";
import { MusicOutput, MusicRail, musicPageModels } from "./pages/music-page";
import { SpeakOutput, SpeakRail, speakPageModels } from "./pages/speak-page";
import { type GenerateBlocker, TtsFooter } from "./pages/tts-workspace";
import { TranscribeOutput, TranscribeRail } from "./pages/transcribe-page";
import { type AudioPickerRow, audioRowMatchesWorkflow } from "./picker-filter";
import { useAudioWorkspaceStore } from "./stores/audio-workspace-store";
import { InstructionsField, MossLanguageField } from "./components/instructions-fields";
import {
  type AudioWorkflowId,
  audioWorkflowTab,
  clipWorkflow,
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
  const pageWorkflow: AudioWorkflowId =
    slotForWorkflow(workflow) === mode
      ? workflow
      : mode === "transcribe"
        ? "transcribe"
        : "speak";
  const ttsWorkflow: "speak" | "music" =
    pageWorkflow === "music" ? "music" : "speak";
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
  const generationPresentation = audioGenerationPresentation(generationPhase);

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
    peaksById,
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
    handleGenerate,
    generationError,
    setGenerationError,
  } = useSpeechGeneration({
    workflow: ttsWorkflow,
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
    loadMoreVisible,
    fallbackClip: pageFallbackClip,
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
  // Send to: only pages that can take a finished clip today. Bytes first, so a failed fetch leaves the page.
  const sendHandlersFor = useCallback(
    (clip: AudioGalleryClip): ClipSendHandlers => ({
      transcribe: () => {
        void (async () => {
          try {
            const blob = await fetchClipBlob(clip.url);
            if (!transitionWorkflow("transcribe")) return;
            const name = `${clip.prompt.trim().slice(0, 40) || "Audio clip"}.wav`;
            handleTranscribeFile(
              new File([blob], name, { type: blob.type || "audio/wav" }),
            );
          } catch (error) {
            toast.error(
              error instanceof Error
                ? error.message
                : "Could not load this clip for transcription.",
            );
          }
        })();
      },
    }),
    [transitionWorkflow, handleTranscribeFile],
  );
  const handleUseTextAgain = useCallback(
    (clip: AudioGalleryClip) => {
      if (transitionWorkflow(clipWorkflow(clip))) setPrompt(clip.prompt);
    },
    [transitionWorkflow, setPrompt],
  );

  const pageModelLoaded =
    ttsLoaded && (ttsWorkflow === "music") === musicGeneration;
  const openSelector = useCallback(() => setSelectorOpen(true), []);
  const chooseModelAction = { label: "Choose a model", onClick: openSelector };
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
                  : "Load a speech model to generate.",
              actions: [chooseModelAction],
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
                : null;
  const generateFailure: GenerateBlocker | null = generationError
    ? {
        reason: /[.!?]$/.test(generationError) ? generationError : `${generationError}.`,
        actions: [{ label: "Choose a different model", onClick: openSelector }],
      }
    : null;
  useEffect(() => {
    setGenerationError(null);
  }, [status?.active_model, setGenerationError]);
  useEffect(() => {
    if (busy === "generating") setGenerationError(null);
  }, [busy, setGenerationError]);
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
  const pageRootRef = useRef<HTMLDivElement | null>(null);
  const generateShortcut = useRef<() => void>(() => {});
  generateShortcut.current = () => {
    if (canGenerate) void handleGenerate();
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
  const instructionsKind = instructionsFieldKind(ttsWorkflow, status?.audio_type);

  const selectorModels =
    mode === "speak"
      ? ttsWorkflow === "music"
        ? musicPageModels(MODELS_BY_MODE.speak, isMac)
        : speakPageModels(MODELS_BY_MODE.speak, isMac)
      : MODELS_BY_MODE[mode];
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
  const selectorValue =
    mode === "speak"
      ? (status?.active_model ?? undefined)
      : (selectedSttRepo ?? undefined);

  const capabilityLine =
    mode === "speak"
      ? ttsLoaded
        ? pageModelLoaded
          ? audioCapabilityLine(musicGeneration ? "music" : "tts", status?.audio_type)
          : musicGeneration
            ? "The loaded model makes music. Pick a speech model."
            : "The loaded model is not a music model."
        : status?.active_model
          ? "The loaded model is not a TTS audio model."
          : ttsWorkflow === "music"
            ? "No music model loaded."
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
                  : ttsWorkflow === "music"
                    ? []
                    : trainedTtsModels
              }
              rowFilter={selectorRowFilter}
              loadedModelIdOverride={
                mode === "transcribe" && sttReady
                  ? (selectedSttRepo ?? undefined)
                  : undefined
              }
              loaded={mode === "transcribe" ? sttReady : undefined}
              value={selectorValue}
              onValueChange={handleModelSelect}
              onEject={busy === null && selectorValue ? handleEject : undefined}
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
              "hover-scrollbar flex min-h-0 flex-1 flex-col px-10 max-sm:px-5 pt-9 pb-12 @[50rem]:overflow-y-auto",
              mode === "speak"
                ? "panel-scroll-fade-action"
                : "panel-scroll-fade",
              settingsFadeClass,
            )}
          >
            {/* One child, so the scroll fades see the rail grow (they watch only the first child): with
                the heading first, opening Advanced left the bottom fade over the last controls. */}
            <div className="flex flex-col gap-4">
              {/* Same heading treatment as the Images and Video Create panes, so the media panes stay level (#7986). */}
              <div className="mb-2 grid gap-1.5">
                <WorkflowTitleMenu
                  workflow={pageWorkflow}
                  onSelect={transitionWorkflow}
                />
                <p className="text-ui-11p5 leading-snug text-muted-foreground">
                  {workflowTab.hint}
                </p>
                {/* The always-on capability line: which task the selected model actually does. */}
                <p className="text-xs leading-snug text-muted-foreground">
                  {capabilityLine}
                </p>
              </div>

              {mode === "speak" ? (
                (() => {
                  const railProps = {
                    prompt,
                    setPrompt,
                    toolPanels: instructionsKind ? (
                      <>
                        <InstructionsField
                          instructionsKind={instructionsKind}
                          musicNeedsDescription={musicNeedsDescription}
                          audioInstructions={audioInstructions}
                          setAudioInstructions={setAudioInstructions}
                        />
                        {instructionsKind === "style" ? (
                          <MossLanguageField
                            audioLanguage={audioLanguage}
                            setAudioLanguage={setAudioLanguage}
                          />
                        ) : null}
                      </>
                    ) : null,
                    audioDevice,
                    busy,
                    isRecording,
                    status,
                    setAudioDeviceState,
                    ttsLoaded,
                    handleEject,
                    samplingControls,
                    audioOptionSpecs,
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
          </div>
          {mode === "speak" ? (
            /* The scroll mask provides the fade; leave the footer unpainted to avoid dark-mode banding. */
            <div className="relative z-10 flex shrink-0 justify-center px-10 pt-0.5 pb-4">
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
                  workflow: ttsWorkflow,
                  peaksById,
                  sendHandlersFor,
                  pending:
                    busy === "generating" && generationPresentation
                      ? {
                          title:
                            prompt.trim() ||
                            audioInstructions.trim() ||
                            (ttsWorkflow === "music" ? "New track" : "New clip"),
                          status: generationPresentation.status,
                          elapsedSeconds,
                          canStop: generationPresentation.canStop,
                          onStop: handleStopGeneration,
                        }
                      : null,
                  clips: visibleClips,
                  selectedClip,
                  selectedClipSrc,
                  srcById,
                  handleDownloadClip,
                  handleDeleteClip,
                  fallbackClip: pageFallbackClip,
                  handleDownloadFallbackClip,
                  handleClearGallery: handleClearWorkflowGallery,
                  historyReorder,
                  hasMore,
                  loadMore: loadMoreVisible,
                  selectedId,
                  selectClip,
                  handleTogglePin,
                  active,
                  handleArchiveClip,
                  handleDownloadClipById,
                  onUseTextAgain: handleUseTextAgain,
                  handleCopyPrompt,
                  freshClipId,
                  onFreshClipFocused: clearFreshClip,
                  announcement,
                };
                return ttsWorkflow === "music" ? (
                  <MusicOutput {...outputProps} modelReady={pageModelLoaded} />
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
