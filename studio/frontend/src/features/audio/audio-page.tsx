// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Audio page: Generate (TTS via the main inference slot) and Transcribe (STT
// via the dictation sidecar). Training lives on the Train page. The page stays
// mounted across tab switches (see __root.tsx), so `active` gates polling,
// popovers and the recorder rather than lifecycle.

import { TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import {
  AudioWave01Icon,
  Copy01Icon,
  Delete02Icon,
  Download01Icon,
  Mic01Icon,
  MoreVerticalIcon,
  SparklesIcon,
  StopIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import { AdvancedDisclosure } from "@/components/advanced-disclosure";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { usePlatformStore } from "@/config/env";
import {
  type InferenceStatusResponse,
  ParamSlider,
  getInferenceStatus,
  listGgufVariants,
  loadModel,
  unloadModel,
} from "@/features/chat";
import {
  SttModelNotDownloadedError,
  cancelSttLoad,
  fetchSttStatus,
  loadSttModel,
  startSttDownload,
  sttEngineStatusFor,
  transcribeAudioBlob,
  unloadSttModel,
} from "@/features/chat/adapters/studio-model-dictation-adapter";
import { useStagedDownload } from "@/features/hub/download-manager";
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { ModelSelector } from "@/features/model-picker/components/model-selector";
import { AUDIO_CATALOG } from "@/features/model-picker/components/model-selector/model-catalog";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import type {
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker/components/model-selector/types";
import {
  isTrackingSttDownload,
  trackSttDownload,
} from "@/features/settings/lib/stt-download-mirror";
import { sttModelSize } from "@/features/settings/stores/stt-model-catalog";
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
import { useScrollFades } from "@/hooks/use-scroll-fades";
import { BlobUrlCache } from "@/lib/blob-url-cache";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useNavigate, useSearch } from "@tanstack/react-router";

import {
  type AudioGalleryClip,
  clearAudioGallery,
  deleteAudioClip,
  fetchClipObjectUrl,
  generateAudio,
  listAudioGallery,
} from "./api";
import {
  type AudioBusy,
  type SttDownloadedArtifact,
  canTransitionAudioMode,
  exactGgufLoadSelector,
  expectedGgufDownloadBytes,
  isTtsAudioType,
  macTtsPickAction,
  micStreamRequestIsCurrent,
  persistedClipForGeneration,
  reconcileSttSelection,
  resolveAudioPickTask,
  resolveSttResidency,
  selectAutoGgufVariant,
  stagedTtsLoadIsOwned,
  sttDownloadedArtifacts,
  sttSelectionReady,
} from "./audio-page-policy";
import {
  audioCapabilityLine,
  audioModelsForTask,
  audioTaskFor,
  ggufSiblingFor,
  macTtsCatalogChoiceIsRunnable,
  sttEngineForRepoId,
  sttRepoIdForSidecarKey,
  sttSidecarKeyFor,
} from "./catalog";

const MODELS_BY_MODE: Record<CreateMode, ModelOption[]> = {
  speak: audioModelsForTask("tts"),
  transcribe: audioModelsForTask("stt"),
};

function deviceSizeBytes(label: string): number {
  const match = label.trim().match(/^(\d+(?:\.\d+)?)\s*(MB|GB)$/i);
  if (!match) return 0;
  const value = Number(match[1]);
  return value * (match[2].toUpperCase() === "GB" ? 1024 ** 3 : 1024 ** 2);
}
const HUB_TASKS_BY_MODE = {
  speak: ["text-to-speech"],
  transcribe: ["automatic-speech-recognition"],
} as const;

const PAGE_SIZE = 50;
// WAV clips run a few MB a minute; 64 MB keeps a healthy scrollback resident.
const CLIP_BLOB_BUDGET_BYTES = 64 * 1024 * 1024;

// Module scope so a tab switch re-renders the gallery instantly (the page stays mounted, but a
// remount after an unlikely unmount still reuses fetched clips).
const galleryCache: {
  clips: AudioGalleryClip[];
  hasMore: boolean;
  nextOffset: number;
  selectedId: string | null;
  srcById: BlobUrlCache;
} = {
  clips: [],
  hasMore: false,
  nextOffset: 0,
  selectedId: null,
  srcById: new BlobUrlCache(CLIP_BLOB_BUDGET_BYTES),
};

type CreateMode = "speak" | "transcribe";

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div className="grid gap-1.5">
      <span className="text-ui-13 font-medium text-foreground">{label}</span>
      {children}
      {hint ? (
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          {hint}
        </p>
      ) : null}
    </div>
  );
}

/** Per-row actions for a history clip, in a dots menu so rows keep one line.
 *  Mirrors the model rows' MoreVertical pattern. */
function ClipRowMenu({
  clip,
  onDownload,
  onCopyPrompt,
  onUseAsText,
  onDelete,
}: {
  clip: AudioGalleryClip;
  onDownload: () => void;
  onCopyPrompt: () => void;
  onUseAsText: () => void;
  onDelete: () => void;
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          onClick={(event) => event.stopPropagation()}
          aria-label={`Actions for ${clip.prompt || "clip"}`}
          // Hidden until the row is hovered or the menu is open, so a long list
          // stays quiet; keyboard focus reveals it too.
          className="flex size-5 shrink-0 items-center justify-center rounded-md text-muted-foreground/60 opacity-0 transition-colors hover:bg-black/5 hover:text-foreground focus-visible:opacity-100 group-hover:opacity-100 data-[state=open]:opacity-100 dark:hover:bg-white/10"
        >
          <HugeiconsIcon
            icon={MoreVerticalIcon}
            strokeWidth={1.75}
            className="size-3.5"
          />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        <DropdownMenuItem onSelect={onUseAsText}>
          <HugeiconsIcon icon={SparklesIcon} className="size-4" />
          Use text again
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={onCopyPrompt}>
          <HugeiconsIcon icon={Copy01Icon} className="size-4" />
          Copy text
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={onDownload}>
          <HugeiconsIcon icon={Download01Icon} className="size-4" />
          Download WAV
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem variant="destructive" onSelect={onDelete}>
          <HugeiconsIcon icon={Delete02Icon} className="size-4" />
          Delete
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function formatClipDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0:00";
  const whole = Math.round(seconds);
  const minutes = Math.floor(whole / 60);
  const rest = whole % 60;
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

export function AudioPage({ active = true }: { active?: boolean }) {
  const [mode, setMode] = useState<CreateMode>("speak");
  const [selectorOpen, setSelectorOpen] = useState(false);
  const [busy, setBusy] = useState<AudioBusy>(null);
  const busyRef = useRef<AudioBusy>(busy);
  busyRef.current = busy;

  // --- TTS (main inference slot) -----------------------------------------
  const [status, setStatus] = useState<InferenceStatusResponse | null>(null);
  const [prompt, setPrompt] = useState("");
  const [temperature, setTemperature] = useState(0.6);
  const [maxTokens, setMaxTokens] = useState(2048);
  const generateAbort = useRef<AbortController | null>(null);
  const ttsLoadInFlight = useRef(false);
  const ttsStatusRefreshGeneration = useRef(0);
  const ttsLoadGeneration = useRef(0);
  const pendingTtsLoad = useRef<{
    generation: number;
    repoId: string;
    controller: AbortController;
    requestStarted: boolean;
  } | null>(null);

  // --- STT (dictation sidecar) -------------------------------------------
  const [selectedSttRepo, setSelectedSttRepo] = useState<string | null>(null);
  const [sttLoadedModel, setSttLoadedModel] = useState<string | null>(null);
  const [sttLoadedEngine, setSttLoadedEngine] = useState<
    "transformers" | "gguf" | "mtmd" | null
  >(null);
  const [downloadedSttArtifacts, setDownloadedSttArtifacts] = useState<
    SttDownloadedArtifact[]
  >([]);
  const [transcript, setTranscript] = useState("");
  const [transcribedName, setTranscribedName] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [micRequestPending, setMicRequestPending] = useState(false);
  /** Audio the server produced but could not persist; kept so the generation is
   *  not lost when the gallery write fails. Cleared once a real clip lands. */
  const [fallbackClip, setFallbackClip] = useState<{
    url: string;
    prompt: string;
    model: string;
  } | null>(null);
  const fallbackClipRef = useRef(fallbackClip);
  fallbackClipRef.current = fallbackClip;
  const loadingMoreRef = useRef(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordStreamRef = useRef<MediaStream | null>(null);
  const discardRecordingRef = useRef(false);
  const selectedSttRepoRef = useRef<string | null>(selectedSttRepo);
  selectedSttRepoRef.current = selectedSttRepo;
  const activeRef = useRef(active);
  activeRef.current = active;
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const sttStatusRefreshGeneration = useRef(0);
  const sttLoadGeneration = useRef(0);
  const sttLoadingGeneration = useRef<number | null>(null);
  const deferredSttLoad = useRef<{
    repoId: string;
    sidecarKey: string;
    engine: "transformers" | "gguf" | "mtmd";
  } | null>(null);
  const ttsPickGeneration = useRef(0);
  const ttsInspectionGeneration = useRef<number | null>(null);
  const stagedTtsGeneration = useRef(0);
  const pendingStagedTtsLoad = useRef<{
    repoId: string;
    ggufFilename: string;
    generation: number;
  } | null>(null);
  const stagedTtsLoadDeferred = useRef(false);
  const micRequestGeneration = useRef(0);
  const micPendingGeneration = useRef<number | null>(null);
  const transcriptionAbort = useRef<AbortController | null>(null);

  const stopRecordStream = useCallback(() => {
    for (const track of recordStreamRef.current?.getTracks() ?? [])
      track.stop();
    recordStreamRef.current = null;
  }, []);
  const stopAndDiscardRecording = useCallback(() => {
    // Also invalidates a getUserMedia request that has not resolved yet. Its
    // eventual stream is stopped before a MediaRecorder can be created.
    micRequestGeneration.current += 1;
    micPendingGeneration.current = null;
    setMicRequestPending(false);
    const recorder = recorderRef.current;
    if (recorder) {
      discardRecordingRef.current = true;
      if (recorder.state !== "inactive") recorder.stop();
      recorderRef.current = null;
      setIsRecording(false);
    }
    stopRecordStream();
  }, [stopRecordStream]);

  // --- Gallery ------------------------------------------------------------
  const [clips, setClips] = useState<AudioGalleryClip[]>(galleryCache.clips);
  const [hasMore, setHasMore] = useState(galleryCache.hasMore);
  const [selectedId, setSelectedId] = useState<string | null>(
    galleryCache.selectedId,
  );
  const [srcById, setSrcById] = useState<Record<string, string>>(
    galleryCache.srcById.toRecord(),
  );

  const {
    attach: attachSettingsScroll,
    onScroll: onSettingsScroll,
    className: settingsFadeClass,
  } = useScrollFades();
  const [advancedOpen, setAdvancedOpen] = usePersistedToggle(
    "unsloth_audio_advanced_open",
  );

  const refreshStatus = useCallback(async () => {
    const generation = ++ttsStatusRefreshGeneration.current;
    try {
      const next = await getInferenceStatus();
      if (generation !== ttsStatusRefreshGeneration.current) return;
      setStatus(next);
    } catch {
      if (generation !== ttsStatusRefreshGeneration.current) return;
      // Do not leave Generate enabled against residency the backend can no
      // longer confirm. A later refresh adopts the recovered runtime.
      setStatus(null);
    }
  }, []);

  const refreshSttStatus = useCallback(async () => {
    const generation = ++sttStatusRefreshGeneration.current;
    try {
      const selectedRepo = selectedSttRepoRef.current;
      const selectedKey = selectedRepo ? sttSidecarKeyFor(selectedRepo) : null;
      const selectedEngine = selectedRepo
        ? sttEngineForRepoId(selectedRepo)
        : null;
      const stt = await fetchSttStatus(
        undefined,
        selectedEngine === "transformers"
          ? (selectedKey ?? undefined)
          : undefined,
      );
      if (generation !== sttStatusRefreshGeneration.current) return;
      const nextDownloadedArtifacts = sttDownloadedArtifacts(
        stt,
        sttRepoIdForSidecarKey,
      );
      setDownloadedSttArtifacts((current) =>
        current.length === nextDownloadedArtifacts.length &&
        current.every(
          (artifact, index) =>
            artifact.repoId === nextDownloadedArtifacts[index].repoId &&
            artifact.sidecarKey === nextDownloadedArtifacts[index].sidecarKey &&
            artifact.engine === nextDownloadedArtifacts[index].engine,
        )
          ? current
          : nextDownloadedArtifacts,
      );
      const selectedBlock = selectedKey
        ? (sttEngineStatusFor(stt, selectedKey, selectedEngine ?? undefined) ??
          (selectedEngine === "transformers" ? stt : undefined))
        : undefined;
      const preservePending = Boolean(
        selectedBlock?.loading ||
          sttLoadingGeneration.current !== null ||
          deferredSttLoad.current !== null,
      );
      const residency = resolveSttResidency(
        stt,
        selectedEngine,
        preservePending,
      );
      const loadedModel = residency?.model ?? null;
      setSttLoadedModel(loadedModel);
      setSttLoadedEngine(residency?.engine ?? null);
      const reconciled = reconcileSttSelection({
        selectedRepo,
        loadedModel,
        loadedEngine: residency?.engine,
        preservePending,
        sidecarKeyFor: sttSidecarKeyFor,
        repoIdForSidecarKey: sttRepoIdForSidecarKey,
        engineForRepo: sttEngineForRepoId,
      });
      selectedSttRepoRef.current = reconciled;
      setSelectedSttRepo(reconciled);
    } catch {
      if (generation !== sttStatusRefreshGeneration.current) return;
      setSttLoadedModel(null);
      setSttLoadedEngine(null);
    }
  }, []);

  const ensureClipSrc = useCallback(async (clip: AudioGalleryClip) => {
    const cached = galleryCache.srcById.get(clip.id);
    if (cached) {
      galleryCache.srcById.touch(clip.id);
      return;
    }
    try {
      const fetched = await fetchClipObjectUrl(clip.url);
      galleryCache.srcById.set(clip.id, fetched.url, fetched.bytes);
      galleryCache.srcById.prune(
        galleryCache.selectedId ? [galleryCache.selectedId] : [],
      );
      setSrcById(galleryCache.srcById.toRecord());
    } catch {
      // Clip may have been deleted server-side; the next gallery refresh drops it.
    }
  }, []);

  const refreshGallery = useCallback(async () => {
    try {
      const page = await listAudioGallery(0, PAGE_SIZE);
      galleryCache.clips = page.audio;
      galleryCache.hasMore = page.has_more;
      galleryCache.nextOffset = page.audio.length;
      setClips(page.audio);
      setHasMore(page.has_more);
      if (
        galleryCache.selectedId &&
        !page.audio.some((c) => c.id === galleryCache.selectedId)
      ) {
        galleryCache.selectedId = page.audio[0]?.id ?? null;
        setSelectedId(galleryCache.selectedId);
      }
      if (
        !galleryCache.selectedId &&
        !fallbackClipRef.current &&
        page.audio.length > 0
      ) {
        galleryCache.selectedId = page.audio[0].id;
        setSelectedId(galleryCache.selectedId);
      }
    } catch {
      // Same recoverable-poll stance as status.
    }
  }, []);

  const loadMore = useCallback(async () => {
    // Repeated scroll events near the bottom would otherwise each fire with the
    // same offset and append the same page, duplicating clips and React keys.
    if (loadingMoreRef.current) return;
    loadingMoreRef.current = true;
    try {
      const page = await listAudioGallery(galleryCache.nextOffset, PAGE_SIZE);
      galleryCache.nextOffset += page.audio.length;
      const known = new Set(galleryCache.clips.map((clip) => clip.id));
      galleryCache.clips = [
        ...galleryCache.clips,
        ...page.audio.filter((clip) => !known.has(clip.id)),
      ];
      galleryCache.hasMore = page.has_more;
      setClips(galleryCache.clips);
      setHasMore(page.has_more);
    } catch {
      // Retry on the next scroll.
    } finally {
      loadingMoreRef.current = false;
    }
  }, []);

  // Resync on activation: another tab may have loaded/unloaded models meanwhile.
  useEffect(() => {
    if (!active) return;
    void refreshStatus();
    void refreshSttStatus();
    void refreshGallery();
  }, [active, refreshStatus, refreshSttStatus, refreshGallery]);

  // The selected clip needs its bytes before the player can play it.
  useEffect(() => {
    const clip = clips.find((c) => c.id === selectedId);
    if (clip) void ensureClipSrc(clip);
  }, [clips, selectedId, ensureClipSrc]);

  const selectClip = useCallback((id: string) => {
    galleryCache.selectedId = id;
    setSelectedId(id);
    setFallbackClip(null);
  }, []);

  // --- Model selection ----------------------------------------------------

  const loadTtsModel = useCallback(
    async (repoId: string, ggufFilename?: string | null) => {
      if (ttsLoadInFlight.current) return;
      const generation = ++ttsLoadGeneration.current;
      const controller = new AbortController();
      const pending = {
        generation,
        repoId,
        controller,
        requestStarted: false,
      };
      pendingTtsLoad.current = pending;
      const isCurrent = () =>
        generation === ttsLoadGeneration.current && activeRef.current;
      ttsLoadInFlight.current = true;
      busyRef.current = "loading";
      setBusy("loading");
      const toastId = toast.loading(`Loading ${repoId}…`);
      try {
        const res = await loadModel(
          {
            model_path: repoId,
            hf_token: hfApiToken(getHfToken()) ?? null,
            max_seq_length: 2048,
            load_in_4bit: false,
            is_lora: false,
            gguf_variant: ggufFilename ?? null,
          },
          {
            signal: controller.signal,
            onRequestStart: () => {
              pending.requestStarted = true;
            },
          },
        );
        if (!isCurrent()) return;
        if (res.is_audio && isTtsAudioType(res.audio_type)) {
          toast.success(`Model loaded (${res.audio_type ?? "audio"})`, {
            id: toastId,
          });
        } else {
          toast.error(`${repoId} loaded but is not a supported TTS model.`, {
            id: toastId,
          });
        }
      } catch (error) {
        if (isCurrent()) {
          toast.error(
            error instanceof Error ? error.message : "Model load failed.",
            { id: toastId },
          );
        } else {
          toast.dismiss(toastId);
        }
      } finally {
        if (pendingTtsLoad.current?.generation === generation)
          pendingTtsLoad.current = null;
        ttsLoadInFlight.current = false;
        busyRef.current = null;
        setBusy(null);
        if (activeRef.current) void refreshStatus();
      }
    },
    [refreshStatus],
  );

  // Stage uncached Hub GGUFs through the shared manager so Audio gets the same
  // progress, cancellation, resume and disk preflight behavior as Chat/Images/Video.
  const loadTtsModelRef = useRef(loadTtsModel);
  loadTtsModelRef.current = loadTtsModel;
  const invalidatePendingStagedTts = useCallback(() => {
    stagedTtsGeneration.current += 1;
    pendingStagedTtsLoad.current = null;
    stagedTtsLoadDeferred.current = false;
  }, []);
  const transitionMode = useCallback(
    (nextMode: CreateMode) => {
      if (nextMode === mode) {
        if (nextMode === "transcribe") invalidatePendingStagedTts();
        return true;
      }
      if (!canTransitionAudioMode(busyRef.current)) {
        toast.info(
          "Wait for the active audio task to finish before switching modes.",
        );
        return false;
      }

      if (nextMode === "transcribe") invalidatePendingStagedTts();
      if (busyRef.current === "generating") generateAbort.current?.abort();
      stopAndDiscardRecording();
      setMode(nextMode);
      return true;
    },
    [invalidatePendingStagedTts, mode, stopAndDiscardRecording],
  );
  const { stage: stageTtsDownload } = useStagedDownload({
    scopeId: "audio",
    onReady: () => {
      const pending = pendingStagedTtsLoad.current;
      if (
        !pending ||
        !stagedTtsLoadIsOwned(
          pending.generation,
          stagedTtsGeneration.current,
          modeRef.current,
        )
      ) {
        pendingStagedTtsLoad.current = null;
        stagedTtsLoadDeferred.current = false;
        return;
      }
      // Audio stays mounted across tabs. Loading from a hidden page would evict
      // the visible page's model; likewise, wait for an active generation to end.
      if (!active || busyRef.current !== null) {
        stagedTtsLoadDeferred.current = true;
        return;
      }
      pendingStagedTtsLoad.current = null;
      void loadTtsModelRef.current(pending.repoId, pending.ggufFilename);
    },
  });

  useEffect(() => {
    if (!active || busy !== null || !stagedTtsLoadDeferred.current) return;
    stagedTtsLoadDeferred.current = false;
    const pending = pendingStagedTtsLoad.current;
    if (
      !pending ||
      !stagedTtsLoadIsOwned(
        pending.generation,
        stagedTtsGeneration.current,
        modeRef.current,
      )
    ) {
      pendingStagedTtsLoad.current = null;
      return;
    }
    pendingStagedTtsLoad.current = null;
    void loadTtsModelRef.current(pending.repoId, pending.ggufFilename);
  }, [active, busy]);

  const loadOrStageTtsModel = useCallback(
    (
      repoId: string,
      ggufFilename: string | null,
      meta: ModelSelectorChangeMeta,
    ) => {
      if (
        meta.source === "hub" &&
        meta.isDownloaded === false &&
        ggufFilename
      ) {
        const generation = ++stagedTtsGeneration.current;
        pendingStagedTtsLoad.current = { repoId, ggufFilename, generation };
        stagedTtsLoadDeferred.current = false;
        stageTtsDownload([
          {
            repoId,
            files: [ggufFilename],
            bytes: meta.expectedBytes ?? 0,
            ggufFilename,
          },
        ]);
        return;
      }

      // A cached/local/direct pick supersedes any staged auto-load. The manager
      // may keep downloading globally, but its old completion cannot load here.
      invalidatePendingStagedTts();
      stageTtsDownload([]);
      void loadTtsModelRef.current(repoId, ggufFilename);
    },
    [invalidatePendingStagedTts, stageTtsDownload],
  );

  const ensureSttLoaded = useCallback(
    async (
      repoId: string,
      sidecarKey: string,
      engine: "transformers" | "gguf" | "mtmd",
    ) => {
      const generation = ++sttLoadGeneration.current;
      sttLoadingGeneration.current = generation;
      const isCurrent = () =>
        generation === sttLoadGeneration.current &&
        activeRef.current &&
        selectedSttRepoRef.current === repoId;

      setBusy("loading");
      const toastId = toast.loading(`Preparing ${sidecarKey}…`);
      try {
        try {
          await loadSttModel(sidecarKey, engine);
        } catch (error) {
          if (!(error instanceof SttModelNotDownloadedError)) throw error;
          if (!isCurrent()) return;
          await startSttDownload(sidecarKey, hfApiToken(getHfToken()), engine);
          // STT owns its specialized transfer, but the existing mirror gives
          // it the same global Downloads row, progress and Cancel action as
          // every other Studio model download. Do not reset an adopted row.
          if (!isTrackingSttDownload(sidecarKey, engine)) {
            trackSttDownload(sidecarKey, {
              // Audio owns the final load through its active/selection
              // generation guards. The Voice-settings mirror must not warm a
              // stale sidecar behind those guards when this transfer lands.
              warmSelectedVoiceModelOnComplete: false,
              engine,
              repoId,
            });
          }
          // The shared Downloads panel now owns transfer progress. Keep this
          // toast for the short model-load phase after the bytes land.
          toast.dismiss(toastId);
          // A completed download may outlive this page or selection. Re-check
          // ownership around every await so an old pick cannot replace a newer
          // sidecar when its poll finally finishes.
          for (;;) {
            await new Promise((resolve) => setTimeout(resolve, 1000));
            if (!isCurrent()) return;
            const stt = await fetchSttStatus(
              undefined,
              engine === "transformers" ? sidecarKey : undefined,
            );
            if (!isCurrent()) return;
            const block = sttEngineStatusFor(stt, sidecarKey, engine);
            const download = block?.download;
            // Cancel comes from the shared Downloads row. It is terminal for
            // this preparation attempt, not permission to try loading a
            // partial checkpoint.
            if (download?.cancelled) return;
            if (download?.error) throw new Error(download.error);
            if (!download?.downloading) break;
          }
          if (!isCurrent()) return;
          toast.loading(`Loading ${sidecarKey}…`, { id: toastId });
          await loadSttModel(sidecarKey, engine);
        }
        if (isCurrent())
          toast.success("Transcription model ready", { id: toastId });
      } catch (error) {
        if (isCurrent()) {
          toast.error(
            error instanceof Error
              ? error.message
              : "Transcription model failed.",
            { id: toastId },
          );
        }
      } finally {
        if (sttLoadingGeneration.current === generation) {
          sttLoadingGeneration.current = null;
          setBusy(null);
          void refreshSttStatus();
        } else {
          toast.dismiss(toastId);
        }
      }
    },
    [refreshSttStatus],
  );

  // A hidden page may let the shared download continue, but it must not load
  // the sidecar. Returning to the same still-selected repo resumes preparation
  // once; a different selection/eject clears this deferred ownership.
  useEffect(() => {
    if (!active) {
      ttsPickGeneration.current += 1;
      const pending = pendingTtsLoad.current;
      if (pending) {
        ttsLoadGeneration.current += 1;
        pendingTtsLoad.current = null;
        pending.controller.abort();
        if (pending.requestStarted)
          void unloadModel({ model_path: pending.repoId }).catch(() => {});
      }
      if (ttsInspectionGeneration.current !== null) {
        ttsInspectionGeneration.current = null;
        busyRef.current = null;
        setBusy(null);
      }

      if (sttLoadingGeneration.current !== null) {
        const repoId = selectedSttRepoRef.current;
        deferredSttLoad.current = repoId
          ? {
              repoId,
              sidecarKey: sttSidecarKeyFor(repoId),
              engine: sttEngineForRepoId(repoId),
            }
          : null;
        sttLoadGeneration.current += 1;
        sttLoadingGeneration.current = null;
        busyRef.current = null;
        setBusy(null);
        const engine = repoId ? sttEngineForRepoId(repoId) : null;
        if (engine) void cancelSttLoad(engine).catch(() => {});
      }
      return;
    }

    const deferred = deferredSttLoad.current;
    deferredSttLoad.current = null;
    if (deferred && selectedSttRepoRef.current === deferred.repoId) {
      void (async () => {
        try {
          const status = await fetchSttStatus(
            undefined,
            deferred.engine === "transformers"
              ? deferred.sidecarKey
              : undefined,
          );
          const download = sttEngineStatusFor(
            status,
            deferred.sidecarKey,
            deferred.engine,
          )?.download;
          if (download?.cancelled && download.model === deferred.sidecarKey)
            return;
        } catch {
          // Status is advisory here; the normal preparation path reports errors.
        }
        if (activeRef.current && selectedSttRepoRef.current === deferred.repoId)
          void ensureSttLoaded(
            deferred.repoId,
            deferred.sidecarKey,
            deferred.engine,
          );
      })();
    }
  }, [active, ensureSttLoaded]);

  const isMac = usePlatformStore((s) => s.deviceType) === "mac";

  const handleModelSelect = useCallback(
    async (id: string, meta: ModelSelectorChangeMeta) => {
      if (busyRef.current !== null) return;
      // Selecting a different artifact while recording is a lifecycle change
      // even when it stays in Transcribe mode; never let the old capture submit
      // against a sidecar that this pick is replacing.
      stopAndDiscardRecording();
      deferredSttLoad.current = null;
      const selectionGeneration = ++ttsPickGeneration.current;
      // Catalog first; an uncurated Hub pick falls back to its pipeline tag, or
      // every community ASR repo would load into the TTS slot.
      const task = resolveAudioPickTask(audioTaskFor(id), meta.pipelineTag);
      if (task === "stt") {
        // An STT pick owns Transcribe: it runs on the sidecar, not the main slot.
        if (!transitionMode("transcribe")) return;
        const sidecarKey = sttSidecarKeyFor(id);
        const engine = sttEngineForRepoId(id);
        deferredSttLoad.current = null;
        selectedSttRepoRef.current = id;
        setSelectedSttRepo(id);
        void ensureSttLoaded(id, sidecarKey, engine);
        return;
      }
      // TTS (or an uncurated repo the user pasted, which /load will validate).
      if (!transitionMode("speak")) return;
      const exactGguf = exactGgufLoadSelector(meta);
      const isGguf = Boolean(
        meta.isGguf ||
          exactGguf ||
          /(?:^|[-/])gguf(?:$|[-/])/i.test(id) ||
          id.toLowerCase().endsWith(".gguf"),
      );
      const ggufSibling = isGguf ? null : ggufSiblingFor(id);
      const macAction = macTtsPickAction({ isMac, isGguf, ggufSibling });
      if (macAction === "reject") {
        toast.error(
          `${id} has no runnable GGUF TTS build. MLX cannot generate text-to-speech from its safetensors checkpoint on this Mac.`,
          { duration: 7000 },
        );
        return;
      }
      if (macAction === "use-gguf-sibling" && ggufSibling) {
        toast.info(
          `Loading the GGUF build of ${id}. MLX has no text-to-speech decoder, so the safetensors build cannot generate on this Mac.`,
          { duration: 7000 },
        );
        // Resolving the sibling is part of the model load lifecycle. Reserve
        // the slot so Generate cannot run the old resident model and then be
        // evicted by this inspection's eventual completion.
        ttsInspectionGeneration.current = selectionGeneration;
        busyRef.current = "loading";
        setBusy("loading");
        try {
          const listing = await listGgufVariants(
            ggufSibling,
            hfApiToken(getHfToken()),
          );
          if (selectionGeneration !== ttsPickGeneration.current) return;
          const variant = selectAutoGgufVariant(
            listing.variants,
            listing.default_variant,
          );
          if (!variant) {
            toast.error(
              `${ggufSibling} does not publish a runnable GGUF file.`,
            );
            return;
          }
          if (ttsInspectionGeneration.current === selectionGeneration) {
            ttsInspectionGeneration.current = null;
            busyRef.current = null;
            setBusy(null);
          }
          loadOrStageTtsModel(ggufSibling, variant.filename, {
            ...meta,
            source: "hub",
            isGguf: true,
            ggufFilename: variant.filename,
            ggufVariant: variant.quant,
            isDownloaded: variant.downloaded === true && !variant.partial,
            expectedBytes: expectedGgufDownloadBytes(variant),
          });
        } catch (error) {
          if (selectionGeneration !== ttsPickGeneration.current) return;
          toast.error(
            error instanceof Error
              ? error.message
              : `Could not inspect ${ggufSibling}.`,
          );
        } finally {
          if (ttsInspectionGeneration.current === selectionGeneration) {
            ttsInspectionGeneration.current = null;
            busyRef.current = null;
            setBusy(null);
          }
        }
        return;
      }
      loadOrStageTtsModel(id, exactGguf, meta);
    },
    [
      ensureSttLoaded,
      isMac,
      loadOrStageTtsModel,
      stopAndDiscardRecording,
      transitionMode,
    ],
  );

  // A pick handed over from the chat model selector arrives as ?model= (+ ?quant= and task).
  const navigateSelf = useNavigate();
  const routeSearch = useSearch({ strict: false }) as {
    model?: string;
    quant?: string;
    task?: string;
  };
  const handledRouteModel = useRef<string | null>(null);
  useEffect(() => {
    if (!active) return;
    const wanted = routeSearch.model;
    if (!wanted) {
      handledRouteModel.current = null;
      return;
    }
    const key = `${wanted}|${routeSearch.quant ?? ""}|${routeSearch.task ?? ""}`;
    if (handledRouteModel.current === key) return;
    // The persistent Audio page may still be finishing hidden work. Keep the
    // handoff in the URL and retry it when that work releases the lifecycle.
    if (busyRef.current !== null) return;
    handledRouteModel.current = key;
    handleModelSelect(wanted, {
      source: "hub",
      isLora: false,
      ggufFilename: routeSearch.quant ?? undefined,
      // Chat-to-Audio routing cannot preserve the inventory flag, so stage the
      // exact forwarded GGUF. An already-cached job completes immediately.
      isDownloaded: routeSearch.quant ? false : undefined,
      pipelineTag: routeSearch.task ?? null,
    });
    void navigateSelf({ to: "/audio", search: {}, replace: true });
  }, [
    active,
    busy,
    routeSearch.model,
    routeSearch.quant,
    routeSearch.task,
    handleModelSelect,
    navigateSelf,
  ]);

  // --- Speak --------------------------------------------------------------

  const ttsLoaded = Boolean(
    status?.active_model &&
      isTtsAudioType(status.audio_type, status.is_gguf === true),
  );
  const handleEject = useCallback(() => {
    if (busy !== null || isRecording) {
      toast.info("Stop the active audio task before ejecting its model.");
      return;
    }

    // Eject also owns unresolved permission requests. Invalidating here makes
    // their eventual streams self-discard instead of recording for an old STT pick.
    stopAndDiscardRecording();

    if (mode === "transcribe") {
      const selected = selectedSttRepo;
      if (!selected) return;
      const selectedEngine = sttEngineForRepoId(selected);

      // A selection can exist before its sidecar is resident. Only issue an
      // unload when our latest sidecar snapshot says this selection owns it.
      if (
        sttLoadedModel !== sttSidecarKeyFor(selected) ||
        sttLoadedEngine !== selectedEngine
      ) {
        deferredSttLoad.current = null;
        selectedSttRepoRef.current = null;
        sttLoadGeneration.current += 1;
        setSelectedSttRepo(null);
        return;
      }

      setBusy("unloading");
      const toastId = toast.loading("Unloading transcription model…");
      void (async () => {
        try {
          await unloadSttModel(selectedEngine);
          deferredSttLoad.current = null;
          selectedSttRepoRef.current = null;
          sttLoadGeneration.current += 1;
          setSelectedSttRepo(null);
          await refreshSttStatus();
          toast.success("Transcription model unloaded", {
            id: toastId,
            duration: 1200,
          });
        } catch (error) {
          toast.error(
            error instanceof Error
              ? error.message
              : "Failed to unload transcription model.",
            { id: toastId },
          );
        } finally {
          setBusy(null);
        }
      })();
      return;
    }

    const activeModel = status?.active_model;
    if (!activeModel) return;

    // An old managed completion must not immediately replace the model the
    // user just ejected. The global download may continue for later use.
    invalidatePendingStagedTts();
    stageTtsDownload([]);

    setBusy("unloading");
    const toastId = toast.loading("Unloading model…");
    void (async () => {
      try {
        // Non-forced unload is deliberate: the backend refuses rather than
        // killing an active Chat or API generation owned by another surface.
        await unloadModel({ model_path: activeModel });
        await refreshStatus();
        toast.success("Model unloaded", { id: toastId, duration: 1200 });
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : "Failed to unload model.",
          { id: toastId },
        );
      } finally {
        setBusy(null);
      }
    })();
  }, [
    busy,
    isRecording,
    mode,
    refreshStatus,
    refreshSttStatus,
    selectedSttRepo,
    invalidatePendingStagedTts,
    stageTtsDownload,
    status?.active_model,
    sttLoadedModel,
    sttLoadedEngine,
    stopAndDiscardRecording,
  ]);

  const handleGenerate = useCallback(async () => {
    const text = prompt.trim();
    if (!text) return;
    setBusy("generating");
    const controller = new AbortController();
    generateAbort.current = controller;
    try {
      const generated = await generateAudio(text, {
        temperature,
        max_tokens: maxTokens,
        signal: controller.signal,
      });
      await refreshGallery();
      const generatedClip = persistedClipForGeneration(
        generated.clip_id,
        galleryCache.clips,
      );
      if (generatedClip) {
        setFallbackClip(null);
        selectClip(generatedClip.id);
      } else {
        // Gallery persistence is best-effort server-side, so a full or unwritable
        // disk still returns the audio. Play it from the response rather than
        // dropping an expensive generation on the floor.
        galleryCache.selectedId = null;
        setSelectedId(null);
        setFallbackClip({
          url: `data:audio/wav;base64,${generated.audio.data}`,
          prompt: text,
          model: generated.model,
        });
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        toast.error(
          error instanceof Error ? error.message : "Audio generation failed.",
        );
        await refreshStatus();
      }
    } finally {
      generateAbort.current = null;
      setBusy(null);
    }
  }, [
    prompt,
    temperature,
    maxTokens,
    refreshGallery,
    refreshStatus,
    selectClip,
  ]);

  const handleStopGeneration = useCallback(() => {
    generateAbort.current?.abort();
  }, []);

  // Leaving the page keeps the tab mounted, so cancel on both deactivation and unmount.
  useEffect(() => {
    if (!active) generateAbort.current?.abort();
  }, [active]);
  useEffect(() => () => generateAbort.current?.abort(), []);

  // --- Transcribe ---------------------------------------------------------

  const sttSelected = selectedSttRepo !== null;
  const sttReady = sttSelectionReady(
    selectedSttRepo,
    sttLoadedModel,
    sttSidecarKeyFor,
    selectedSttRepo ? sttEngineForRepoId(selectedSttRepo) : null,
    sttLoadedEngine,
  );
  const runTranscription = useCallback(
    async (blob: Blob, name: string) => {
      if (!selectedSttRepo) return;
      const key = sttSidecarKeyFor(selectedSttRepo);
      const engine = sttEngineForRepoId(selectedSttRepo);
      if (sttLoadedModel !== key || sttLoadedEngine !== engine) {
        toast.info("Wait for the transcription model to finish loading.");
        return;
      }
      transcriptionAbort.current?.abort();
      const controller = new AbortController();
      transcriptionAbort.current = controller;
      setBusy("transcribing");
      setTranscribedName(name);
      try {
        const text = await transcribeAudioBlob(blob, {
          model: key,
          engine,
          language: "",
          signal: controller.signal,
        });
        if (controller.signal.aborted || !activeRef.current) return;
        setTranscript(text);
        if (!text) toast.info("The model heard no speech in that audio.");
      } catch (error) {
        if (controller.signal.aborted) return;
        toast.error(
          error instanceof Error ? error.message : "Transcription failed.",
        );
      } finally {
        if (transcriptionAbort.current === controller) {
          transcriptionAbort.current = null;
          setBusy(null);
          if (activeRef.current) void refreshSttStatus();
        }
      }
    },
    [selectedSttRepo, sttLoadedModel, sttLoadedEngine, refreshSttStatus],
  );

  const handleRecordToggle = useCallback(async () => {
    if (isRecording) {
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") recorder.stop();
      return;
    }
    if (micPendingGeneration.current !== null) return;
    const requestGeneration = ++micRequestGeneration.current;
    micPendingGeneration.current = requestGeneration;
    setMicRequestPending(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      if (
        !micStreamRequestIsCurrent(
          requestGeneration,
          micRequestGeneration.current,
          activeRef.current,
        )
      ) {
        for (const track of stream.getTracks()) track.stop();
        return;
      }
      recordStreamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      });
      recorder.addEventListener("stop", () => {
        const discard = discardRecordingRef.current;
        discardRecordingRef.current = false;
        setIsRecording(false);
        stopRecordStream();
        recorderRef.current = null;
        const blob = new Blob(chunks, {
          type: recorder.mimeType || "audio/webm",
        });
        if (!discard && blob.size > 0) void runTranscription(blob, "Recording");
      });
      recorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
    } catch {
      // getUserMedia may have succeeded even if MediaRecorder construction or
      // start failed. Release that acquired stream instead of leaving the mic
      // live with no recorder UI capable of stopping it.
      recorderRef.current = null;
      setIsRecording(false);
      stopRecordStream();
      if (
        micStreamRequestIsCurrent(
          requestGeneration,
          micRequestGeneration.current,
          activeRef.current,
        )
      )
        toast.error("Could not access the microphone.");
    } finally {
      if (micPendingGeneration.current === requestGeneration) {
        micPendingGeneration.current = null;
        setMicRequestPending(false);
      }
    }
  }, [isRecording, runTranscription, stopRecordStream]);

  // Release the microphone on unmount AND whenever the page goes inactive: the
  // page stays mounted across tab switches, so unmount alone left a hidden
  // recorder capturing until the user came back and stopped it.
  useEffect(() => {
    if (!active) {
      stopAndDiscardRecording();
      transcriptionAbort.current?.abort();
    }
    return () => {
      stopAndDiscardRecording();
      transcriptionAbort.current?.abort();
    };
  }, [active, stopAndDiscardRecording]);

  const handleTranscribeFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      void runTranscription(file, file.name);
    },
    [runTranscription],
  );

  const handleCopyTranscript = useCallback(() => {
    void navigator.clipboard.writeText(transcript).then(
      () => toast.success("Transcript copied"),
      () => toast.error("Could not copy the transcript."),
    );
  }, [transcript]);

  const handleDownloadTranscript = useCallback(() => {
    const blob = new Blob([transcript], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${(transcribedName ?? "transcript").replace(/\.[^.]+$/, "")}.txt`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [transcript, transcribedName]);

  // --- Gallery actions ----------------------------------------------------

  const handleDeleteClip = useCallback(
    async (id: string) => {
      try {
        await deleteAudioClip(id);
        galleryCache.srcById.delete(id);
        setSrcById(galleryCache.srcById.toRecord());
        await refreshGallery();
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : "Could not delete the clip.",
        );
      }
    },
    [refreshGallery],
  );

  const handleClearGallery = useCallback(async () => {
    try {
      await clearAudioGallery();
      galleryCache.srcById.clear();
      galleryCache.selectedId = null;
      setSrcById({});
      setSelectedId(null);
      await refreshGallery();
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Could not clear the gallery.",
      );
    }
  }, [refreshGallery]);

  const handleDownloadClip = useCallback(
    (clip: AudioGalleryClip) => {
      const src = srcById[clip.id];
      if (!src) return;
      const anchor = document.createElement("a");
      anchor.href = src;
      anchor.download = `${clip.id}.wav`;
      anchor.click();
    },
    [srcById],
  );

  const handleDownloadFallbackClip = useCallback(() => {
    if (!fallbackClip) return;
    const anchor = document.createElement("a");
    anchor.href = fallbackClip.url;
    anchor.download = "generated-audio.wav";
    anchor.click();
  }, [fallbackClip]);

  /** Download from a history row, whose bytes are only fetched once selected. */
  const handleDownloadClipById = useCallback(async (clip: AudioGalleryClip) => {
    let temporaryUrl: string | null = null;
    try {
      let src = galleryCache.srcById.get(clip.id);
      if (!src) {
        const fetched = await fetchClipObjectUrl(clip.url);
        src = fetched.url;
        temporaryUrl = fetched.url;
      }
      const anchor = document.createElement("a");
      anchor.href = src;
      anchor.download = `${clip.id}.wav`;
      anchor.click();
    } catch {
      toast.error("Could not download the clip.");
    } finally {
      // A history-row download does not need to become resident playback state.
      // Revoke it after the browser has consumed the synthetic click instead of
      // bypassing the persistent gallery cache's 64 MB budget.
      if (temporaryUrl) {
        const url = temporaryUrl;
        window.setTimeout(() => URL.revokeObjectURL(url), 0);
      }
    }
  }, []);

  const handleCopyPrompt = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success("Text copied");
    } catch {
      toast.error("Could not copy the text.");
    }
  }, []);

  // --- Render -------------------------------------------------------------

  const selectedClip = clips.find((c) => c.id === selectedId) ?? null;
  const selectorModels =
    mode === "speak" && isMac
      ? MODELS_BY_MODE.speak.filter((model) =>
          macTtsCatalogChoiceIsRunnable(model.id),
        )
      : MODELS_BY_MODE[mode];
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
        ? audioCapabilityLine("tts", status?.audio_type)
        : status?.active_model
          ? "The loaded model is not a TTS audio model."
          : "No TTS model loaded."
      : sttSelected
        ? audioCapabilityLine("stt", sttReady ? "ready" : "loading")
        : "No transcription model selected.";

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden pt-[var(--studio-content-top-inset,0px)]">
      <div className="relative flex h-[48px] shrink-0 items-start justify-between pl-[var(--studio-media-header-left-inset,1.5rem)] pr-2 pt-[var(--studio-chat-header-padding-top,11px)]">
        <div className="flex items-center gap-2">
          <ModelSelector
            models={selectorModels}
            additionalOnDeviceModels={
              mode === "transcribe" ? sttOnDeviceModels : undefined
            }
            loadedModelIdOverride={
              mode === "transcribe" && sttReady
                ? (selectedSttRepo ?? undefined)
                : undefined
            }
            value={selectorValue}
            onValueChange={handleModelSelect}
            onEject={busy === null && selectorValue ? handleEject : undefined}
            variant="ghost"
            className="!h-[34px]"
            task={HUB_TASKS_BY_MODE[mode]}
            catalog={AUDIO_CATALOG}
            // TTS/ASR come from the checkpoint's own tokenizer, not a curated
            // recipe, so any publisher's audio repo loads here.
            communityModelPolicy="search-only"
            placeholder="Select audio model"
            open={active && selectorOpen}
            onOpenChange={(o) => setSelectorOpen(active && o)}
          />
        </div>
        <div className="pointer-events-none absolute inset-x-0 top-[var(--studio-chat-header-padding-top,11px)] flex justify-center">
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
            className="pointer-events-auto h-[34px] [&>button]:h-[34px] [&>button]:px-11"
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
        </div>
      </div>
      <div className="flex min-h-0 w-full min-w-0 flex-1 overflow-hidden pl-2 pr-5 pt-9 sm:pr-8">
        <div className="relative flex w-[408px] shrink-0 flex-col overflow-hidden border-r border-border/60 pl-8">
          <div
            ref={attachSettingsScroll}
            onScroll={onSettingsScroll}
            className={cn(
              "hover-scrollbar panel-scroll-fade flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto pb-20 pl-0.5 pr-8",
              settingsFadeClass,
            )}
          >
            {/* Same heading treatment as the Images and Video Create panes, so
                  the media panes stay level (#7986). */}
            <div className="mb-2 grid gap-1.5">
              <h2 className="flex items-center gap-2 font-heading text-xl font-medium leading-none text-foreground">
                <HugeiconsIcon
                  icon={mode === "speak" ? AudioWave01Icon : Mic01Icon}
                  className="size-[18px] shrink-0"
                />
                {mode === "speak" ? "Generate audio" : "Transcribe"}
              </h2>
              {/* The always-on capability line: which task the selected model actually does. */}
              <p className="text-xs leading-snug text-muted-foreground">
                {capabilityLine}
              </p>
            </div>

            <PillTabs
              ariaLabel="Create mode"
              value={mode}
              onValueChange={(v) => transitionMode(v as CreateMode)}
              fit={true}
              className="h-[30px] self-start [&>button]:h-[30px] [&>button]:px-6"
              tabs={[
                { value: "speak", label: "Generate" },
                { value: "transcribe", label: "Transcribe" },
              ]}
            />

            {mode === "speak" ? (
              <>
                <Field
                  label="Text"
                  hint="What the model should say. Generation runs on the loaded TTS model and lands in the gallery on the right."
                >
                  <Textarea
                    value={prompt}
                    onChange={(event) => setPrompt(event.target.value)}
                    placeholder="Type the sentence to speak…"
                    className="min-h-28"
                  />
                </Field>
                <AdvancedDisclosure
                  open={advancedOpen}
                  onOpenChange={setAdvancedOpen}
                  description="Generation sampling. Changes apply to the next audio clip."
                >
                  <ParamSlider
                    label="Temperature"
                    value={temperature}
                    min={0}
                    max={1.5}
                    step={0.05}
                    onChange={setTemperature}
                  />
                  <ParamSlider
                    label="Max tokens"
                    value={maxTokens}
                    min={256}
                    max={8192}
                    step={256}
                    onChange={setMaxTokens}
                  />
                </AdvancedDisclosure>
              </>
            ) : (
              <>
                <Field
                  label="Microphone"
                  hint="Record a clip and it is transcribed when you stop."
                >
                  <Button
                    variant={isRecording ? "destructive" : "secondary"}
                    disabled={
                      (!isRecording && (!sttReady || busy !== null)) ||
                      micRequestPending
                    }
                    onClick={handleRecordToggle}
                  >
                    <HugeiconsIcon
                      icon={isRecording ? StopIcon : Mic01Icon}
                      className="mr-2 size-4"
                    />
                    {isRecording
                      ? "Stop recording"
                      : micRequestPending
                        ? "Waiting for microphone…"
                        : "Record"}
                  </Button>
                </Field>
                <Field
                  label="Audio file"
                  hint="Or transcribe an existing recording (wav, mp3, m4a, webm…)."
                >
                  <input
                    type="file"
                    accept="audio/*"
                    disabled={
                      !sttReady ||
                      busy !== null ||
                      isRecording ||
                      micRequestPending
                    }
                    onChange={(event) => {
                      handleTranscribeFile(event.target.files?.[0]);
                      event.target.value = "";
                    }}
                    className="text-ui-13 file:mr-3 file:rounded-md file:border-0 file:bg-muted file:px-3 file:py-1.5 file:text-ui-13 file:font-medium"
                  />
                </Field>
                {sttSelected ? null : (
                  <p className="text-ui-11p5 leading-snug text-muted-foreground">
                    Pick a speech-to-text model (Whisper or Qwen3-ASR) from the
                    selector above to transcribe.
                  </p>
                )}
              </>
            )}
          </div>
          {mode === "speak" ? (
            <div className="pointer-events-none absolute inset-x-0 bottom-0 flex justify-center pb-7 pl-8 pr-8">
              <Button
                className="btn-float-action pointer-events-auto h-11 px-8 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100"
                onClick={
                  busy === "generating" ? handleStopGeneration : handleGenerate
                }
                disabled={
                  busy === "generating"
                    ? false
                    : busy !== null || !ttsLoaded || !prompt.trim()
                }
                variant={busy === "generating" ? "destructive" : "default"}
              >
                {busy === "generating" ? (
                  <>
                    <HugeiconsIcon icon={StopIcon} className="mr-2 size-4" />
                    Stop
                  </>
                ) : (
                  "Generate"
                )}
              </Button>
            </div>
          ) : null}
        </div>

        <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden pl-2">
          {mode === "transcribe" ? (
            <div className="hover-scrollbar flex flex-1 flex-col gap-3 overflow-auto p-6 pl-8">
              {busy === "transcribing" ? (
                <div className="flex items-center gap-2 text-ui-13 text-muted-foreground">
                  <Spinner className="size-4" />
                  Transcribing {transcribedName ?? "audio"}…
                </div>
              ) : null}
              {transcript ? (
                <>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={handleCopyTranscript}
                    >
                      Copy
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={handleDownloadTranscript}
                    >
                      <HugeiconsIcon
                        icon={Download01Icon}
                        className="mr-2 size-3.5"
                      />
                      Download .txt
                    </Button>
                  </div>
                  <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
                    {transcript}
                  </p>
                </>
              ) : busy !== "transcribing" ? (
                <p className="text-ui-13 text-muted-foreground">
                  The transcript appears here. It is not stored: copy or
                  download what you want to keep.
                </p>
              ) : null}
            </div>
          ) : (
            <div className="flex min-h-0 flex-1 flex-col gap-4 p-6 pl-8">
              <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-4">
                {selectedClip ? (
                  <div className="flex w-full max-w-xl flex-col gap-3">
                    <p className="line-clamp-2 text-ui-13 text-muted-foreground">
                      {selectedClip.prompt}
                    </p>
                    {/* Auth-protected bytes, so the element plays the fetched object URL. */}
                    <audio
                      controls={true}
                      src={srcById[selectedClip.id]}
                      className="w-full"
                    />
                    <div className="flex items-center gap-2 text-ui-11p5 text-muted-foreground">
                      <span>{selectedClip.model}</span>
                      <span>·</span>
                      <span>{formatClipDuration(selectedClip.duration_s)}</span>
                      <span className="flex-1" />
                      <Button
                        variant="ghost"
                        size="sm"
                        aria-label="Download audio clip"
                        disabled={!srcById[selectedClip.id]}
                        onClick={() => handleDownloadClip(selectedClip)}
                      >
                        <HugeiconsIcon
                          icon={Download01Icon}
                          className="size-3.5"
                        />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        aria-label="Delete audio clip"
                        onClick={() => void handleDeleteClip(selectedClip.id)}
                      >
                        <HugeiconsIcon
                          icon={Delete02Icon}
                          className="size-3.5"
                        />
                      </Button>
                    </div>
                  </div>
                ) : fallbackClip ? (
                  <div className="flex w-full max-w-xl flex-col gap-3">
                    <p className="line-clamp-2 text-ui-13 text-muted-foreground">
                      {fallbackClip.prompt}
                    </p>
                    <audio
                      controls={true}
                      src={fallbackClip.url}
                      className="w-full"
                    />
                    <div className="flex items-center gap-2 text-ui-11p5 text-muted-foreground">
                      <span>
                        {fallbackClip.model} · not saved to the gallery
                      </span>
                      <span className="flex-1" />
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleDownloadFallbackClip}
                      >
                        <HugeiconsIcon
                          icon={Download01Icon}
                          className="size-3.5"
                        />
                        Download WAV
                      </Button>
                    </div>
                  </div>
                ) : (
                  <p className="text-ui-13 text-muted-foreground">
                    Generated speech lands here. Load a TTS model, type a
                    sentence, and press Generate.
                  </p>
                )}
              </div>
              {clips.length > 0 ? (
                <div className="flex shrink-0 flex-col gap-2">
                  <div className="flex items-center justify-between">
                    <span className="text-ui-11p5 font-medium text-muted-foreground">
                      History
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => void handleClearGallery()}
                    >
                      Clear all
                    </Button>
                  </div>
                  <div
                    className="hover-scrollbar flex max-h-40 flex-col gap-1 overflow-y-auto"
                    onScroll={(event) => {
                      const el = event.currentTarget;
                      if (
                        hasMore &&
                        el.scrollTop + el.clientHeight >= el.scrollHeight - 40
                      ) {
                        void loadMore();
                      }
                    }}
                  >
                    {clips.map((clip) => (
                      // Shell, not a button: the dots menu is a button and cannot nest.
                      <div
                        key={clip.id}
                        className={cn(
                          "group flex items-center rounded-md pr-1 transition-colors hover:bg-muted",
                          clip.id === selectedId && "bg-muted",
                        )}
                      >
                        <button
                          type="button"
                          onClick={() => selectClip(clip.id)}
                          className="flex min-w-0 flex-1 items-center gap-2 px-2 py-1.5 text-left text-ui-13"
                        >
                          <HugeiconsIcon
                            icon={AudioWave01Icon}
                            className="size-3.5 shrink-0 text-muted-foreground"
                          />
                          <span className="min-w-0 flex-1 truncate">
                            {clip.prompt}
                          </span>
                          <span className="shrink-0 text-ui-11p5 text-muted-foreground">
                            {formatClipDuration(clip.duration_s)}
                          </span>
                        </button>
                        <ClipRowMenu
                          clip={clip}
                          onDownload={() => void handleDownloadClipById(clip)}
                          onCopyPrompt={() =>
                            void handleCopyPrompt(clip.prompt)
                          }
                          onUseAsText={() => {
                            if (transitionMode("speak")) setPrompt(clip.prompt);
                          }}
                          onDelete={() => void handleDeleteClip(clip.id)}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
