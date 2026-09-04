// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Audio page: Generate (TTS via the main inference slot) and Transcribe (STT via the
// dictation sidecar). The page stays mounted across tab switches, so `active` gates polling,
// popovers and the recorder rather than lifecycle.

import { TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import {
  Archive02Icon,
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
  useMemo,
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
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { usePlatformStore } from "@/config/env";
import {
  type InferenceStatusResponse,
  ParamSlider,
  cancelPreStreamRunReservations,
  confirmStopRunningChatsIfNeeded,
  getInferenceStatus,
  listGgufVariants,
  listLoras,
  loadModel,
  requestLocalPromptQueueStop,
  unloadModel,
  useChatRuntimeStore,
} from "@/features/chat";
import {
  PcmRecorder,
  type SegmentRecorder,
  createAudioRecorder,
} from "@/features/chat/adapters/pcm-recorder";
import {
  SttModelNotDownloadedError,
  StudioModelDictationAdapter,
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
import { confirmRemoteCodeIfNeeded } from "@/features/security";
import { useSettingsDialogStore } from "@/features/settings";
import {
  isTrackingSttDownload,
  trackSttDownload,
} from "@/features/settings/lib/stt-download-mirror";
import { sttModelSize } from "@/features/settings/stores/stt-model-catalog";
import { usePersistedChoice } from "@/hooks/use-persisted-choice";
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
import { useScrollFades } from "@/hooks/use-scroll-fades";
import { fetchSystemInfo } from "@/hooks/use-system";
import { BlobUrlCache } from "@/lib/blob-url-cache";
import { subscribeGalleryChanged } from "@/lib/gallery-flags";
import { subscribeModelLifecycle } from "@/lib/model-lifecycle-events";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useNavigate, useSearch } from "@tanstack/react-router";

import {
  type AudioGalleryClip,
  clearAudioGallery,
  deleteAudioClip,
  fetchClipObjectUrl,
  generateAudio,
  getAudioDownloadPlan,
  listAudioGallery,
  setAudioClipFlags,
} from "./api";
import {
  type AudioBusy,
  type AudioGenerationPhase,
  MINIMAX_MUSIC_DEFAULT_SECONDS,
  MINIMAX_MUSIC_FRAMES_PER_SECOND,
  MINIMAX_MUSIC_MAX_SECONDS,
  MOSS_TTS_DEFAULT_SECONDS,
  MOSS_TTS_FRAMES_PER_SECOND,
  MOSS_TTS_MAX_FRAMES,
  type SttDownloadedArtifact,
  audioGenerationPresentation,
  canTransitionAudioMode,
  exactGgufLoadSelector,
  expectedGgufDownloadBytes,
  isGgufTtsTarget,
  isTtsAudioType,
  macTtsPickAction,
  mergeGalleryPage,
  micStreamRequestIsCurrent,
  minimaxMusicFramesForSeconds,
  mossTtsFramesForSeconds,
  mossTtsMaxFrames,
  nativeAudioInstructionsKind,
  persistedClipForGeneration,
  reconcileSttSelection,
  resolveAudioPickTask,
  resolveSttResidency,
  selectAutoGgufVariant,
  stagedTtsLoadIsOwned,
  sttDownloadedArtifacts,
  sttSelectionReady,
  trainedTtsCheckpointIsLoadable,
  trainedTtsCheckpointIsRunnableOnMac,
} from "./audio-page-policy";
import {
  audioCapabilityLine,
  audioModelRequiresRemoteCode,
  audioModelsForTask,
  audioTaskFor,
  ggufSiblingFor,
  isMusicGenerationModel,
  macTtsCatalogChoiceIsRunnable,
  sttEngineForRepoId,
  sttRepoIdForSidecarKey,
  sttSidecarKeyFor,
  usesNativeAudioRuntime,
} from "./catalog";

const MODELS_BY_MODE: Record<CreateMode, ModelOption[]> = {
  speak: audioModelsForTask("tts"),
  transcribe: audioModelsForTask("stt"),
};

/** What to call a model on screen. A Hub repo is its id; a checkpoint trained here is an output
 *  directory, and the full path in a toast reads as a bug. */
function audioModelLabel(id: string): string {
  if (!/^(?:[a-zA-Z]:[\\/]|[\\/]|~)/.test(id)) return id;
  const leaf = id.split(/[\\/]/).filter(Boolean).pop() ?? id;
  // Training stamps the output directory with an epoch; it means nothing to a reader.
  return leaf.replace(/_\d{10,}$/, "");
}

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
// The list route clamps a page to 200, so asking for more silently gets 200 back.
const MAX_PAGE_SIZE = 200;
// Mirrors the STT sidecar's own limits, so a recording is stopped at the boundary rather than uploaded and refused.
// The two it mirrors are _MAX_AUDIO_SECONDS and STT_AUDIO_B64_MAX_CHARS.
const RECORDING_MAX_SECONDS = 30 * 60;
// STT_AUDIO_RAW_MAX_BYTES in utils/upload_limits.py. A larger client cap let a dense codec build
// a recording the raw route then refused with 413.
const RECORDING_MAX_BYTES = 25 * 1024 * 1024;
const RECORDING_CHUNK_MS = 1000;
const TTS_MAX_TOKENS = 8192;
// Max tokens caps the OUTPUT and the prompt's own tokens sit in the same context window, so
// loading at exactly TTS_MAX_TOKENS made the advertised maximum unreachable.
const TTS_PROMPT_CONTEXT_RESERVE = 2048;
// WAV clips run a few MB a minute; 64 MB keeps a healthy scrollback resident.
const CLIP_BLOB_BUDGET_BYTES = 64 * 1024 * 1024;

// Module scope so a tab switch re-renders the gallery instantly.
const galleryCache: {
  clips: AudioGalleryClip[];
  hasMore: boolean;
  nextCursor: { mtime: number; id: string } | null;
  selectedId: string | null;
  srcById: BlobUrlCache;
} = {
  clips: [],
  hasMore: false,
  nextCursor: null,
  selectedId: null,
  srcById: new BlobUrlCache(CLIP_BLOB_BUDGET_BYTES),
};

type CreateMode = "speak" | "transcribe";
type RemoteCodeApproval = {
  trustRemoteCode: true;
  approvedRemoteCodeFingerprint: string | null;
};

function Field({
  label,
  hint,
  htmlFor,
  children,
}: {
  label: string;
  hint?: string;
  htmlFor: string;
  children: ReactNode;
}) {
  return (
    <div className="grid gap-1.5">
      <label
        className="text-ui-13 font-medium text-foreground"
        htmlFor={htmlFor}
      >
        {label}
      </label>
      {children}
      {hint ? (
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          {hint}
        </p>
      ) : null}
    </div>
  );
}

/** Per-row actions for a history clip, in a dots menu so rows keep one line. Mirrors the model
 *  rows' MoreVertical pattern. */
function ClipRowMenu({
  clip,
  onDownload,
  onCopyPrompt,
  onUseAsText,
  onArchive,
  onDelete,
}: {
  clip: AudioGalleryClip;
  onDownload: () => void;
  onCopyPrompt: () => void;
  onUseAsText: () => void;
  onArchive: () => void;
  onDelete: () => void;
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          onClick={(event) => event.stopPropagation()}
          aria-label={`Actions for ${clip.prompt || "clip"}`}
          // Hidden until the row is hovered or the menu is open, so a long list stays quiet; keyboard
          // focus reveals it too.
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
        <DropdownMenuItem onSelect={onArchive}>
          <HugeiconsIcon icon={Archive02Icon} className="size-4" />
          Archive
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

export function AudioPage({
  active = true,
  onInitialReady,
}: {
  active?: boolean;
  onInitialReady?: () => void;
}) {
  const initialReadySent = useRef(false);
  const [mode, setMode] = useState<CreateMode>("speak");
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
  const [prompt, setPrompt] = useState("");
  const [audioInstructions, setAudioInstructions] = useState("");
  const [audioLanguage, setAudioLanguage] = useState("");
  const [temperature, setTemperature] = useState(0.6);
  // Sending temperature unconditionally puts it in the request's model_fields_set, which the
  // backend reads as an explicit client override that beats the per-model recommendation, so
  // only send it once the user has moved the slider.
  // Spark-TTS wants 0.8, OuteTTS 0.4, so only send it once the user has moved the slider.
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
  const generateAbort = useRef<AbortController | null>(null);
  const handleStopGeneration = useCallback(() => {
    const controller = generateAbort.current;
    if (!controller || controller.signal.aborted) return;
    updateGenerationPhase("stopping");
    controller.abort();
  }, [updateGenerationPhase]);
  const ttsLoadInFlight = useRef(false);
  // A pick that lost the race with a load still settling. Replayed once it does.
  const pendingRoutedTtsPick = useRef<{
    repoId: string;
    ggufFilename?: string | null;
    loadId?: string | null;
    audioType?: string | null;
    remoteCodeApproval?: RemoteCodeApproval;
    isGguf?: boolean | null;
  } | null>(null);
  const ttsStatusRefreshGeneration = useRef(0);
  const ttsLoadGeneration = useRef(0);
  const pendingTtsLoad = useRef<{
    generation: number;
    repoId: string;
    /** What the load request actually sent, which is what a cancel has to name. */
    loadTarget: string;
    loadRequestId: string;
    controller: AbortController;
    requestStarted: boolean;
  } | null>(null);

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
  const [transcriptError, setTranscriptError] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [micRequestPending, setMicRequestPending] = useState(false);
  /** Safari and other WebKit builds ship no MediaRecorder, and an http LAN origin is not a secure
   *  context, so navigator.mediaDevices is undefined. Same check the chat composer uses. */
  const recordingSupported = useMemo(
    () => StudioModelDictationAdapter.isSupported(),
    [],
  );
  /** Audio the server produced that the gallery is not showing yet: either it could not be
   *  persisted, or this refresh missed it. Kept so the generation is playable either way. */
  const [fallbackClip, setFallbackClip] = useState<{
    url: string;
    prompt: string;
    model: string;
    saved: boolean;
  } | null>(null);
  const fallbackClipRef = useRef(fallbackClip);
  fallbackClipRef.current = fallbackClip;
  const loadingMoreRef = useRef(false);
  const galleryRefreshGeneration = useRef(0);
  const recorderRef = useRef<SegmentRecorder | null>(null);
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
  // Residency is not ownership: the activation resync adopts whatever a sidecar already holds,
  // including a model chat dictation loaded. The identity, not a boolean, since another
  // surface can replace the sidecar's model while Audio is inactive and a bare flag would claim
  // that too. Keyed on the model alone, since a "gguf" pick without whisper-server comes back
  // resident under the Transformers fallback.
  const sttLoadedByThisPage = useRef<string | null>(null);
  const sttLoadAbort = useRef<AbortController | null>(null);
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
    ggufFilename: string | null;
    loadId?: string | null;
    audioType?: string | null;
    remoteCodeApproval?: RemoteCodeApproval;
    isGguf?: boolean | null;
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
    // Also invalidates a getUserMedia request that has not resolved yet; its eventual stream is
    // stopped before a MediaRecorder can be created.
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

  const [clips, setClips] = useState<AudioGalleryClip[]>(galleryCache.clips);
  const [hasMore, setHasMore] = useState(galleryCache.hasMore);
  const [selectedId, setSelectedId] = useState<string | null>(
    galleryCache.selectedId,
  );
  const [srcById, setSrcById] = useState<Record<string, string>>(
    galleryCache.srcById.toRecord(),
  );
  const clipSrcLoads = useRef<Map<string, Promise<void>>>(new Map());
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

  const clearTranscript = useCallback(() => {
    setTranscript("");
    setTranscribedName(null);
    setTranscriptError(null);
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
      // Shared sidecar: an adopted model strands the previous pick's transcript under it. null is the
      // 5-minute idle unload, which must not delete it.
      if (reconciled !== null && reconciled !== selectedSttRepoRef.current)
        clearTranscript();
      selectedSttRepoRef.current = reconciled;
      setSelectedSttRepo(reconciled);
    } catch {
      if (generation !== sttStatusRefreshGeneration.current) return;
      setSttLoadedModel(null);
      setSttLoadedEngine(null);
    }
  }, [clearTranscript]);

  const sttSelected = selectedSttRepo !== null;
  const sttReady = sttSelectionReady(
    selectedSttRepo,
    sttLoadedModel,
    sttSidecarKeyFor,
    selectedSttRepo ? sttEngineForRepoId(selectedSttRepo) : null,
    sttLoadedEngine,
  );

  /** Forget the Transcribe pick, releasing its sidecar when this page owns it. */
  const releaseTranscribeSelection = useCallback(async () => {
    const selected = selectedSttRepoRef.current;
    const claim = sttLoadedByThisPage.current;
    const owned =
      sttReady &&
      selected !== null &&
      claim !== null &&
      claim === sttLoadedModel;
    const forget = () => {
      deferredSttLoad.current = null;
      selectedSttRepoRef.current = null;
      sttLoadGeneration.current += 1;
      sttLoadedByThisPage.current = null;
      clearTranscript();
      setSelectedSttRepo(null);
    };
    if (!owned) {
      forget();
      return;
    }
    // Forget only once the sidecar is actually released: clearing first left a failed unload with the
    // model in VRAM and no Eject to retry with. Scoped to the model this page claimed, since
    // another surface can switch the same engine before the request lands.
    await unloadSttModel(sttEngineForRepoId(selected), claim);
    forget();
    await refreshSttStatus();
  }, [clearTranscript, refreshSttStatus, sttReady, sttLoadedModel]);

  const ensureClipSrc = useCallback(async (clip: AudioGalleryClip) => {
    const cached = galleryCache.srcById.get(clip.id);
    if (cached) {
      galleryCache.srcById.touch(clip.id);
      return;
    }
    const pending = clipSrcLoads.current.get(clip.id);
    if (pending) return pending;
    const load = (async () => {
      try {
        const fetched = await fetchClipObjectUrl(clip.url);
        // A delete can finish while protected bytes are in flight. Do not revive its cache entry after
        // the row is already gone.
        if (!galleryCache.clips.some((candidate) => candidate.id === clip.id)) {
          URL.revokeObjectURL(fetched.url);
          return;
        }
        galleryCache.srcById.set(clip.id, fetched.url, fetched.bytes);
        galleryCache.srcById.prune(
          galleryCache.selectedId ? [galleryCache.selectedId] : [],
        );
        setSrcById(galleryCache.srcById.toRecord());
      } catch {
        // Clip may have been deleted server-side; the next gallery refresh drops it.
        toast.error("Could not load this audio clip. Try selecting it again.");
      }
    })();
    clipSrcLoads.current.set(clip.id, load);
    try {
      await load;
    } finally {
      if (clipSrcLoads.current.get(clip.id) === load) {
        clipSrcLoads.current.delete(clip.id);
      }
    }
  }, []);

  const refreshGallery = useCallback(
    async (
      removedId?: string,
      windowSize = PAGE_SIZE,
    ): Promise<AudioGalleryClip[]> => {
      const generation = ++galleryRefreshGeneration.current;
      const wanted = Math.max(PAGE_SIZE, windowSize);
      const asked = Math.min(wanted, MAX_PAGE_SIZE);
      try {
        const page = await listAudioGallery(0, asked);
        // The caller's own fetch: a generation whose clip persisted must not be told otherwise.
        if (generation !== galleryRefreshGeneration.current) return page.audio;
        // A window past the route's cap cannot be covered in one page, and stitching the old scrollback
        // back on keeps a cursor that starts BELOW it, stranding whatever was restored.
        const { clips: merged, stitched } =
          wanted > asked
            ? { clips: [...page.audio], stitched: false }
            : mergeGalleryPage(
                page.audio,
                galleryCache.clips,
                removedId,
                page.has_more,
              );
        galleryCache.clips = merged;
        // A clip record carries no mtime, so kept scrollback has no cursor; keep the deeper one.
        if (!stitched) {
          galleryCache.hasMore = page.has_more;
          galleryCache.nextCursor =
            page.next_before_mtime !== null && page.next_before_id !== null
              ? { mtime: page.next_before_mtime, id: page.next_before_id }
              : null;
        }
        setClips(merged);
        setHasMore(galleryCache.hasMore);
        // The response audio was kept only until its record showed up: left mounted, deleting the
        // now-visible clip made the "saved, waiting for the gallery" copy reappear.
        if (
          fallbackClipRef.current &&
          galleryCache.selectedId &&
          merged.some((c) => c.id === galleryCache.selectedId)
        ) {
          setFallbackClip(null);
        }
        if (
          galleryCache.selectedId &&
          !merged.some((c) => c.id === galleryCache.selectedId)
        ) {
          galleryCache.selectedId = merged[0]?.id ?? null;
          setSelectedId(galleryCache.selectedId);
        }
        if (
          !galleryCache.selectedId &&
          !fallbackClipRef.current &&
          merged.length > 0
        ) {
          galleryCache.selectedId = merged[0].id;
          setSelectedId(galleryCache.selectedId);
        }
        const selected = merged.find(
          (clip) => clip.id === galleryCache.selectedId,
        );
        if (selected) void ensureClipSrc(selected);
        return merged;
      } catch {
        // Same recoverable-poll stance as status.
        return galleryCache.clips;
      }
    },
    [ensureClipSrc],
  );

  const loadMore = useCallback(async () => {
    // Repeated scroll events near the bottom would otherwise each fire with the same offset and
    // append the same page, duplicating clips and React keys.
    if (loadingMoreRef.current) return;
    loadingMoreRef.current = true;
    const refreshGeneration = galleryRefreshGeneration.current;
    const cursor = galleryCache.nextCursor;
    try {
      const page = await listAudioGallery(0, PAGE_SIZE, cursor);
      if (
        refreshGeneration !== galleryRefreshGeneration.current ||
        cursor !== galleryCache.nextCursor
      )
        return;
      galleryCache.nextCursor =
        page.next_before_mtime !== null && page.next_before_id !== null
          ? { mtime: page.next_before_mtime, id: page.next_before_id }
          : null;
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

  useEffect(() => {
    if (!active) return;
    const refreshWhenVisible = () => {
      if (document.hidden) return;
      void refreshGallery(undefined, galleryCache.clips.length);
    };
    window.addEventListener("focus", refreshWhenVisible);
    document.addEventListener("visibilitychange", refreshWhenVisible);
    return () => {
      window.removeEventListener("focus", refreshWhenVisible);
      document.removeEventListener("visibilitychange", refreshWhenVisible);
    };
  }, [active, refreshGallery]);

  // The selected clip needs its bytes before the player can play it.
  useEffect(() => {
    const clip = clips.find((c) => c.id === selectedId);
    if (clip) void ensureClipSrc(clip);
  }, [clips, selectedId, ensureClipSrc]);

  /** `keepFallback` is for the one case where the id is not in `clips` yet: the server persisted the
   *  clip but this refresh missed it, so the response audio has to stay mounted or the player
   *  falls through to the empty state. */
  const selectClip = useCallback(
    (id: string, keepFallback = false) => {
      galleryCache.selectedId = id;
      setSelectedId(id);
      if (!keepFallback) setFallbackClip(null);
      const clip = galleryCache.clips.find((candidate) => candidate.id === id);
      if (clip) void ensureClipSrc(clip);
    },
    [ensureClipSrc],
  );


  const loadTtsModel = useCallback(
    async (
      repoId: string,
      ggufFilename?: string | null,
      // Where the weights actually are: a row cached in a NON-ACTIVE HF cache is loadable only by its
      // snapshot path, which the picker supplies as meta.loadId, and sending the display repo id
      // instead failed offline or re-downloaded into the active cache.
      loadId?: string | null,
      audioType?: string | null,
      remoteCodeApproval?: RemoteCodeApproval,
      // The catalog's answer: the ids alone miss a GGUF repo that does not spell it.
      isGguf?: boolean | null,
    ) => {
      // A routed pick arriving while a previous load is still tearing down would otherwise be dropped,
      // and the route effect has already cleared ?model=, so replay it from the finally below.
      if (ttsLoadInFlight.current || busyRef.current === "generating") {
        pendingRoutedTtsPick.current = {
          repoId,
          ggufFilename,
          loadId,
          audioType,
          remoteCodeApproval,
          isGguf,
        };
        return;
      }
      // A load stops every chat on the shared llama-server, so ask the way Chat does instead of
      // dead-ending on the backend's 409. Claimed before the await: a routed pick arriving while
      // the dialog is open must queue.
      ttsLoadInFlight.current = true;
      // Chat's gate, held across the question and the load. Without it a queue can materialize while
      // the dialog is open, outside the snapshot the answer was given for.
      const lifecycleLease = useChatRuntimeStore.getState().beginModelLoading();
      if (lifecycleLease === null) {
        ttsLoadInFlight.current = false;
        pendingRoutedTtsPick.current = null;
        toast.info("Wait for the current model to finish loading.");
        return;
      }
      const releaseLifecycle = () =>
        useChatRuntimeStore.getState().endModelLoading(lifecycleLease);
      const stopDecision = await confirmStopRunningChatsIfNeeded();
      if (!stopDecision.proceed) {
        releaseLifecycle();
        ttsLoadInFlight.current = false;
        // Declining refuses the swap, so a queued pick must not reopen the dialog.
        pendingRoutedTtsPick.current = null;
        return;
      }
      // The page can go away while the dialog is open, and pendingTtsLoad is still null then, so the
      // deactivation effect has nothing to abort. Queue it for the activation replay.
      if (!activeRef.current) {
        releaseLifecycle();
        ttsLoadInFlight.current = false;
        pendingRoutedTtsPick.current = {
          repoId,
          ggufFilename,
          loadId,
          audioType,
          remoteCodeApproval,
          isGguf,
        };
        return;
      }
      const generation = ++ttsLoadGeneration.current;
      const controller = new AbortController();
      const loadRequestId = crypto.randomUUID();
      const pending = {
        generation,
        repoId,
        // What the request actually sent. Cancelling under the display id works only when the load target
        // is a standard HF cache snapshot; a pinned directory elsewhere does not match and
        // _cancel_scoped_load_attempt then refuses.
        loadTarget: loadId || repoId,
        loadRequestId,
        controller,
        requestStarted: false,
      };
      pendingTtsLoad.current = pending;
      const isCurrent = () =>
        generation === ttsLoadGeneration.current && activeRef.current;
      ttsLoadInFlight.current = true;
      busyRef.current = "loading";
      setBusy("loading");
      const toastId = toast.loading(`Loading ${audioModelLabel(repoId)}…`);
      try {
        const hfToken = hfApiToken(getHfToken()) ?? null;
        let trustRemoteCode = remoteCodeApproval?.trustRemoteCode ?? false;
        let approvedRemoteCodeFingerprint =
          remoteCodeApproval?.approvedRemoteCodeFingerprint ?? null;
        if (
          audioModelRequiresRemoteCode(repoId, audioType) &&
          !remoteCodeApproval
        ) {
          const approved = await confirmRemoteCodeIfNeeded({
            modelName: loadId || repoId,
            hfToken,
            requiresTrustRemoteCode: true,
            onApprove: (fingerprint) => {
              trustRemoteCode = true;
              approvedRemoteCodeFingerprint = fingerprint;
            },
          });
          if (!approved)
            throw new Error(
              "Custom code approval is required to load this model.",
            );
          if (controller.signal.aborted || !isCurrent()) return;
        }
        const wantsCpu = audioDevice === "cpu";
        const isGgufLoad = isGgufTtsTarget({ repoId, ggufFilename, loadId, isGguf });
        const res = await loadModel(
          {
            model_path: loadId || repoId,
            load_request_id: loadRequestId,
            force_cancel_active: stopDecision.forceCancelActive,
            hf_token: hfToken,
            max_seq_length: TTS_MAX_TOKENS + TTS_PROMPT_CONTEXT_RESERVE,
            load_in_4bit: false,
            is_lora: false,
            gguf_variant: ggufFilename ?? null,
            trust_remote_code: trustRemoteCode,
            approved_remote_code_fingerprint: approvedRemoteCodeFingerprint,
            audio_device: wantsCpu ? "cpu" : "auto",
            // GGUF ignores audio_device: llama.cpp offloads unless told not to.
            // An absent speculative_type resolves to "auto", which may attach a GPU
            // drafter, and the backend then evicts image/video for a CPU load.
            ...(wantsCpu && isGgufLoad
              ? // biome-ignore lint/style/useNamingConvention: API schema
                {
                  gpu_memory_mode: "manual" as const,
                  gpu_layers: 0,
                  speculative_type: "off" as const,
                }
              : {}),
          },
          {
            signal: controller.signal,
            runtime: "tts",
            onRequestStart: () => {
              pending.requestStarted = true;
              // Queued prompts would otherwise start on the model this load replaces. Only once /load is
              // actually going out: loadModel returns without sending when a stored token is invalid, and
              // cancelling earlier threw away accepted sends for a swap that never happened.
              cancelPreStreamRunReservations(stopDecision.preStreamRunTokens);
              requestLocalPromptQueueStop(stopDecision.promptQueueThreadIds);
            },
          },
        );
        if (!isCurrent()) return;
        if (res.is_audio && isTtsAudioType(res.audio_type)) {
          toast.success(`Model loaded (${res.audio_type ?? "audio"})`, {
            id: toastId,
          });
          // Only the native runtime and GGUF can be held in RAM.
          if (
            wantsCpu &&
            !isGgufLoad &&
            !usesNativeAudioRuntime(repoId, res.audio_type)
          ) {
            toast.info(
              "This model does not support CPU RAM yet, so it loaded on the GPU.",
              { duration: 6000 },
            );
          }
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
        if (activeRef.current) await refreshStatus();
        ttsLoadInFlight.current = false;
        // Before the replay below, which needs the gate for its own attempt.
        releaseLifecycle();
        busyRef.current = null;
        setBusy(null);
        // Only while Audio is visible: replaying unconditionally started a load with activeRef already
        // false, which the deactivation effect never saw to cancel, so a hidden page could replace
        // the model Chat had loaded.
        if (activeRef.current) replayQueuedTtsPick();
      }
    },
    [refreshStatus, audioDevice],
  );

  // Stage uncached Hub GGUFs through the shared manager so Audio gets the same progress,
  // cancellation, resume and disk preflight as Chat/Images/Video.
  const loadTtsModelRef = useRef(loadTtsModel);
  loadTtsModelRef.current = loadTtsModel;
  /** Start a pick that lost the race with a load still settling. Visible pages only: a load started
   *  while hidden outlives the deactivation effect that would cancel it. */
  const replayQueuedTtsPick = useCallback(() => {
    const queued = pendingRoutedTtsPick.current;
    if (!queued) return;
    pendingRoutedTtsPick.current = null;
    void loadTtsModelRef.current(
      queued.repoId,
      queued.ggufFilename,
      queued.loadId,
      queued.audioType,
      queued.remoteCodeApproval,
      queued.isGguf,
    );
  }, []);
  const invalidatePendingStagedTts = useCallback(() => {
    stagedTtsGeneration.current += 1;
    pendingStagedTtsLoad.current = null;
    stagedTtsLoadDeferred.current = false;
  }, []);
  const invalidatePendingTtsSelection = useCallback(() => {
    ttsPickGeneration.current += 1;
    pendingRoutedTtsPick.current = null;
    invalidatePendingStagedTts();
  }, [invalidatePendingStagedTts]);
  const transitionMode = useCallback(
    (nextMode: CreateMode) => {
      if (nextMode === mode) {
        if (nextMode === "transcribe") invalidatePendingTtsSelection();
        return true;
      }
      if (
        !canTransitionAudioMode(busyRef.current, generationPhaseRef.current)
      ) {
        toast.info(
          "Wait for the active audio task to finish before switching modes.",
        );
        return false;
      }

      if (nextMode === "transcribe") invalidatePendingTtsSelection();
      if (busyRef.current === "generating") handleStopGeneration();
      stopAndDiscardRecording();
      setMode(nextMode);
      // Held through Generate, the sidecar keeps a dictation model in VRAM beside the speech one.
      if (mode === "transcribe") {
        // Resolves to whether the sidecar is actually gone: swallowing the rejection made a failed unload
        // look like a release, so the speech load went ahead with the dictation model and OOMed.
        const release = releaseTranscribeSelection().then(
          () => true,
          (error) => {
            toast.error(
              error instanceof Error
                ? error.message
                : "Failed to release the transcription model.",
            );
            return false;
          },
        );
        // Recorded so a TTS load can wait for the teardown: allocating while the sidecar still holds its
        // model is what OOMs a device that fits either one alone.
        pendingTranscribeRelease.current = release;
        void release.then((released) => {
          if (pendingTranscribeRelease.current !== release) return;
          pendingTranscribeRelease.current = null;
          // The sidecar is still holding its model, so the page must not sit in Speak claiming otherwise:
          // back to Transcribe, where Eject can retry. Only if nothing has moved on since.
          if (!released && modeRef.current === "speak") setMode("transcribe");
        });
      }
      return true;
    },
    [
      invalidatePendingTtsSelection,
      handleStopGeneration,
      mode,
      releaseTranscribeSelection,
      stopAndDiscardRecording,
    ],
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
      // Audio stays mounted across tabs, so loading from a hidden page would evict the visible page's
      // model; likewise, wait for an active generation to end.
      if (!active || busyRef.current !== null) {
        stagedTtsLoadDeferred.current = true;
        return;
      }
      pendingStagedTtsLoad.current = null;
      void loadTtsModelRef.current(
        pending.repoId,
        pending.ggufFilename,
        pending.loadId,
        pending.audioType,
        pending.remoteCodeApproval,
        pending.isGguf,
      );
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
    void loadTtsModelRef.current(
      pending.repoId,
      pending.ggufFilename,
      pending.loadId,
      pending.audioType,
      pending.remoteCodeApproval,
      pending.isGguf,
    );
  }, [active, busy]);

  const loadOrStageTtsModel = useCallback(
    async (
      repoId: string,
      ggufFilename: string | null,
      meta: ModelSelectorChangeMeta,
    ) => {
      const generation = ++stagedTtsGeneration.current;
      pendingStagedTtsLoad.current = null;
      stagedTtsLoadDeferred.current = false;
      stageTtsDownload([]);

      let remoteCodeApproval: RemoteCodeApproval | undefined;
      const hfToken = hfApiToken(getHfToken()) ?? null;
      if (
        meta.source === "hub" &&
        !ggufFilename &&
        audioModelRequiresRemoteCode(repoId, meta.audioType)
      ) {
        try {
          const approved = await confirmRemoteCodeIfNeeded({
            modelName: meta.loadId || repoId,
            hfToken,
            requiresTrustRemoteCode: true,
            onApprove: (fingerprint) => {
              remoteCodeApproval = {
                trustRemoteCode: true,
                approvedRemoteCodeFingerprint: fingerprint,
              };
            },
          });
          if (generation !== stagedTtsGeneration.current) return;
          if (!approved) {
            toast.error("Custom code approval is required to load this model.");
            return;
          }
        } catch (error) {
          if (generation !== stagedTtsGeneration.current) return;
          toast.error(
            error instanceof Error
              ? error.message
              : `Could not verify the code for ${repoId}.`,
          );
          return;
        }
      }

      // Hub TTS picks use the same managed path as Chat. Native models can also depend on a second
      // codec repository, so the selected repo's downloaded badge is not enough: the cache-aware
      // backend plan owns every missing file.
      if (meta.source === "hub" && !ggufFilename) {
        let plan;
        try {
          plan = await getAudioDownloadPlan(
            meta.loadId || repoId,
            hfToken ?? undefined,
          );
        } catch (error) {
          if (generation !== stagedTtsGeneration.current) return;
          if (meta.isDownloaded === true) {
            void loadTtsModelRef.current(
              repoId,
              ggufFilename,
              meta.loadId,
              meta.audioType,
              remoteCodeApproval,
              meta.isGguf,
            );
            return;
          }
          toast.error(
            error instanceof Error
              ? error.message
              : `Could not prepare the download for ${repoId}.`,
          );
          return;
        }
        if (generation !== stagedTtsGeneration.current) return;
        const plannedEntries = plan.entries;
        if (plannedEntries.length > 0) {
          pendingStagedTtsLoad.current = {
            repoId,
            ggufFilename,
            loadId: meta.loadId,
            audioType: meta.audioType,
            remoteCodeApproval,
            isGguf: meta.isGguf,
            generation,
          };
          stageTtsDownload(
            plannedEntries.map((entry) => ({
              repoId: entry.repo_id,
              files: entry.files,
              bytes: entry.bytes,
              ggufFilename: entry.gguf_filename,
              checkpoint: entry.checkpoint,
            })),
          );
          return;
        }
      }

      if (
        meta.source === "hub" &&
        meta.isDownloaded === false &&
        ggufFilename
      ) {
        pendingStagedTtsLoad.current = {
          repoId,
          ggufFilename,
          loadId: meta.loadId,
          audioType: meta.audioType,
          remoteCodeApproval,
          isGguf: meta.isGguf,
          generation,
        };
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

      // A cached/local/direct pick supersedes any staged auto-load: the manager may keep downloading
      // globally, but its old completion cannot load here.
      void loadTtsModelRef.current(
        repoId,
        ggufFilename,
        meta.loadId,
        meta.audioType,
        remoteCodeApproval,
        meta.isGguf,
      );
    },
    [stageTtsDownload],
  );

  const ensureSttLoaded = useCallback(
    async (
      repoId: string,
      sidecarKey: string,
      engine: "transformers" | "gguf" | "mtmd",
    ) => {
      const generation = ++sttLoadGeneration.current;
      const controller = new AbortController();
      sttLoadingGeneration.current = generation;
      sttLoadAbort.current = controller;
      const isCurrent = () =>
        generation === sttLoadGeneration.current &&
        activeRef.current &&
        selectedSttRepoRef.current === repoId;

      setBusy("loading");
      // Ownership is claimed only once the requested model is actually resident: claiming it up front
      // meant a cancelled download left the flag set while the backend kept the previous model, so
      // leaving Transcribe unloaded another surface's model.
      const toastId = toast.loading(`Preparing ${sidecarKey}…`);
      try {
        try {
          await loadSttModel(sidecarKey, engine, controller.signal);
          sttLoadedByThisPage.current = sidecarKey;
        } catch (error) {
          if (!(error instanceof SttModelNotDownloadedError)) throw error;
          if (!isCurrent()) return;
          await startSttDownload(sidecarKey, hfApiToken(getHfToken()), engine);
          // STT owns its specialized transfer, but the existing mirror gives it the same global Downloads
          // row, progress and Cancel as every other model download. Do not reset an adopted row.
          if (!isTrackingSttDownload(sidecarKey, engine)) {
            trackSttDownload(sidecarKey, {
              // Audio owns the final load through its active/selection generation guards, so the
              // Voice-settings mirror must not warm a stale sidecar behind those guards.
              warmSelectedVoiceModelOnComplete: false,
              engine,
              repoId,
            });
          }
          // The shared Downloads panel now owns transfer progress; keep this toast for the short model-load
          // phase after the bytes land.
          toast.dismiss(toastId);
          // A completed download may outlive this page or selection, so re-check ownership around every
          // await or an old pick could replace a newer sidecar.
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
            // Cancel comes from the shared Downloads row. It is terminal for this preparation attempt, not
            // permission to load a partial checkpoint.
            if (download?.cancelled) return;
            if (download?.error) throw new Error(download.error);
            if (!download?.downloading) break;
          }
          if (!isCurrent()) return;
          toast.loading(`Loading ${sidecarKey}…`, { id: toastId });
          await loadSttModel(sidecarKey, engine, controller.signal);
          sttLoadedByThisPage.current = sidecarKey;
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
          await refreshSttStatus();
          if (
            generation === sttLoadGeneration.current &&
            sttLoadingGeneration.current === null
          ) {
            setBusy(null);
          }
        } else {
          toast.dismiss(toastId);
        }
        if (sttLoadAbort.current === controller) sttLoadAbort.current = null;
      }
    },
    [refreshSttStatus],
  );

  // A hidden page may let the shared download continue, but it must not load the sidecar. Returning
  // to the same still-selected repo resumes preparation once; a different selection clears it.
  useEffect(() => {
    if (!active) {
      ttsPickGeneration.current += 1;
      const pending = pendingTtsLoad.current;
      if (pending) {
        ttsLoadGeneration.current += 1;
        pendingTtsLoad.current = null;
        pending.controller.abort();
        if (pending.requestStarted)
          void unloadModel({
            model_path: pending.loadTarget,
            cancel_load_request_id: pending.loadRequestId,
          }).catch(() => {});
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
        sttLoadAbort.current?.abort();
        sttLoadAbort.current = null;
        busyRef.current = null;
        setBusy(null);
      }
      return;
    }

    // A pick queued behind a settling load, held back because the page went away before the load
    // finished. Now that Audio is visible again the attempt is cancellable.
    replayQueuedTtsPick();

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
          // `model` is null once the download thread has stopped, so a cancellation made while this page
          // was hidden matched nothing here and the deferred load restarted the whole download.
          if (
            download?.cancelled &&
            (download.model ?? download.cancelled_model) === deferred.sidecarKey
          )
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
  /** An in-flight Transcribe teardown a following TTS load has to wait behind. */
  const pendingTranscribeRelease = useRef<Promise<boolean> | null>(null);

  const handleModelSelect = useCallback(
    async (id: string, meta: ModelSelectorChangeMeta) => {
      if (busyRef.current !== null) return;
      // Catalog first; an uncurated Hub pick falls back to its pipeline tag, or every community ASR
      // repo would load into the TTS slot.
      const task = resolveAudioPickTask(audioTaskFor(id), meta.pipelineTag);
      const musicPick =
        task !== "stt" && isMusicGenerationModel(id, meta.audioType);
      const selectionGeneration = ++ttsPickGeneration.current;
      if (musicPick) {
        const system = await fetchSystemInfo();
        if (selectionGeneration !== ttsPickGeneration.current) return;
        if (system?.device_backend !== "cuda") {
          toast.error(
            `${id} requires a verified NVIDIA CUDA GPU for local generation.`,
            { duration: 7000 },
          );
          return;
        }
      }
      // Selecting a different artifact while recording is a lifecycle change even in Transcribe mode;
      // never let the old capture submit against a sidecar this pick is replacing.
      stopAndDiscardRecording();
      deferredSttLoad.current = null;
      if (task === "stt") {
        // An STT pick owns Transcribe: it runs on the sidecar, not the main slot.
        if (!transitionMode("transcribe")) return;
        const sidecarKey = sttSidecarKeyFor(id);
        const engine = sttEngineForRepoId(id);
        if (selectedSttRepoRef.current !== id) clearTranscript();
        deferredSttLoad.current = null;
        selectedSttRepoRef.current = id;
        setSelectedSttRepo(id);
        void ensureSttLoaded(id, sidecarKey, engine);
        return;
      }
      // TTS (or an uncurated repo the user pasted, which /load will validate).
      if (!transitionMode("speak")) return;
      // Serialize against a Transcribe release started by that transition.
      const releaseInFlight = pendingTranscribeRelease.current;
      // A release that failed leaves the sidecar resident, so do not stack a speech model on top of
      // it. Back to Transcribe, where Eject can retry.
      // Claimed before the await below, not after: the button only disables on `busy`, so a slow release
      // let several clicks through, each resuming into its own generateAudio while generateAbort
      // tracked only the last.
      if (releaseInFlight && !(await releaseInFlight)) {
        setMode("transcribe");
        return;
      }
      if (ttsPickGeneration.current !== selectionGeneration) return;
      const exactGguf = exactGgufLoadSelector(meta);
      const isGguf = Boolean(
        meta.isGguf || isGgufTtsTarget({ repoId: id, ggufFilename: exactGguf }),
      );
      const ggufSibling = isGguf ? null : ggufSiblingFor(id);
      const nativeRuntime =
        usesNativeAudioRuntime(id, meta.audioType) && !musicPick;
      const macAction = macTtsPickAction({
        isMac,
        isGguf,
        ggufSibling,
        nativeRuntime,
      });
      if (macAction === "reject") {
        toast.error(
          musicPick
            ? `${id} currently requires an NVIDIA CUDA GPU and cannot run locally on this Mac.`
            : `${id} has no runnable GGUF TTS build. MLX cannot generate text-to-speech from its safetensors checkpoint on this Mac.`,
          { duration: 7000 },
        );
        return;
      }
      if (macAction === "use-gguf-sibling" && ggufSibling) {
        toast.info(
          `Loading the GGUF build of ${id}. MLX has no text-to-speech decoder, so the safetensors build cannot generate on this Mac.`,
          { duration: 7000 },
        );
        // Resolving the sibling is part of the model load lifecycle. Reserve the slot so Generate cannot
        // run the old resident model and then be evicted by this inspection's completion.
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
          await loadOrStageTtsModel(ggufSibling, variant.filename, {
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
      await loadOrStageTtsModel(id, exactGguf, meta);
    },
    [
      clearTranscript,
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
    ggufQuant?: string;
    task?: string;
    audioType?: string;
    loadId?: string;
  };
  const handledRouteModel = useRef<string | null>(null);
  useEffect(() => {
    if (!active) return;
    const wanted = routeSearch.model;
    if (!wanted) {
      handledRouteModel.current = null;
      // A task with no model is a mode intent from Settings; without it the page keeps whatever mode it was left in.
      const task = routeSearch.task;
      if (!task) return;
      const intended =
        task === "automatic-speech-recognition" ? "transcribe" : "speak";
      // Left in the URL when the switch is refused, so it retries once busy releases.
      if (intended !== mode && !transitionMode(intended)) return;
      void navigateSelf({ to: "/audio", search: {}, replace: true });
      return;
    }
    const key = `${wanted}|${routeSearch.quant ?? ""}|${routeSearch.ggufQuant ?? ""}|${routeSearch.task ?? ""}|${routeSearch.audioType ?? ""}|${routeSearch.loadId ?? ""}`;
    if (handledRouteModel.current === key) return;
    // The persistent Audio page may still be finishing hidden work, so keep the handoff in the URL
    // and retry it when that work releases the lifecycle.
    if (busyRef.current !== null) return;
    handledRouteModel.current = key;
    handleModelSelect(wanted, {
      source: "hub",
      isLora: false,
      ggufFilename: routeSearch.quant ?? undefined,
      ggufVariant: routeSearch.ggufQuant ?? undefined,
      loadId: routeSearch.loadId ?? undefined,
      audioType: routeSearch.audioType ?? undefined,
      // Chat-to-Audio routing cannot preserve the inventory flag, so stage the exact forwarded GGUF.
      // An already-cached job completes immediately.
      isDownloaded: routeSearch.loadId
        ? true
        : routeSearch.quant
          ? false
          : undefined,
      pipelineTag: routeSearch.task ?? null,
    });
    void navigateSelf({ to: "/audio", search: {}, replace: true });
  }, [
    active,
    busy,
    mode,
    routeSearch.model,
    routeSearch.quant,
    routeSearch.ggufQuant,
    routeSearch.task,
    routeSearch.audioType,
    routeSearch.loadId,
    handleModelSelect,
    navigateSelf,
    transitionMode,
  ]);


  const ttsLoaded = Boolean(
    status?.active_model &&
      isTtsAudioType(status.audio_type, status.is_gguf === true),
  );
  const musicGeneration =
    status?.audio_type === "minimax_music3" ||
    isMusicGenerationModel(status?.active_model);
  const mossLocalGeneration = status?.audio_type === "moss_tts_local";
  const instructionsKind = musicGeneration
    ? "music"
    : nativeAudioInstructionsKind(status?.audio_type);
  const handleEject = useCallback(() => {
    if (busy !== null || isRecording) {
      toast.info("Stop the active audio task before ejecting its model.");
      return;
    }

    // Eject also owns unresolved permission requests. Invalidating here makes their eventual streams
    // self-discard instead of recording for an old STT pick.
    stopAndDiscardRecording();

    if (mode === "transcribe") {
      if (!selectedSttRepo) return;
      // A selection can exist before its sidecar is resident, so an unowned pick is only forgotten.
      if (!sttReady) {
        sttStatusRefreshGeneration.current += 1;
        void releaseTranscribeSelection();
        return;
      }

      setBusy("unloading");
      const toastId = toast.loading("Unloading transcription model…");
      void (async () => {
        try {
          await releaseTranscribeSelection();
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

    // Chat's gate, taken before the question so a queue cannot materialize while the dialog is open
    // and then be stopped by the blanket queue stop.
    const lifecycleLease = useChatRuntimeStore.getState().beginModelLoading();
    if (lifecycleLease === null) {
      toast.info("Wait for the current model to finish loading.");
      return;
    }

    // Busy before the dialog, so a second eject cannot start behind the first.
    setBusy("unloading");
    void (async () => {
      try {
        // Ejecting stops every chat on the shared llama-server, and unforced the backend refused with a
        // 409 the user could only read. Nothing is torn down until the answer is in, so declining
        // leaves the page as it was.
        const stopDecision = await confirmStopRunningChatsIfNeeded(
          "Unloading the model",
          "unload",
        );
        if (!stopDecision.proceed) {
          setBusy(null);
          return;
        }

        // An old managed completion must not immediately replace the model the user just ejected. The
        // global download may continue for later use.
        invalidatePendingStagedTts();
        stageTtsDownload([]);

        const toastId = toast.loading("Unloading model…");
        try {
          cancelPreStreamRunReservations(stopDecision.preStreamRunTokens);
          requestLocalPromptQueueStop(stopDecision.promptQueueThreadIds);
          await unloadModel({
            model_path: activeModel,
            force_cancel_active: stopDecision.forceCancelActive,
          });
          requestLocalPromptQueueStop();
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
      } finally {
        useChatRuntimeStore.getState().endModelLoading(lifecycleLease);
      }
    })();
  }, [
    busy,
    isRecording,
    mode,
    refreshStatus,
    releaseTranscribeSelection,
    selectedSttRepo,
    invalidatePendingStagedTts,
    stageTtsDownload,
    status?.active_model,
    sttReady,
    stopAndDiscardRecording,
  ]);

  const handleGenerate = useCallback(async () => {
    const text = prompt.trim();
    if (!text) return;
    // Same gate the TTS load path uses: switching straight from Transcribe with a speech model
    // already resident needs no load, so nothing else waits for the sidecar teardown, and
    // generating beside a dictation model OOMs a device that fits either alone. Claimed before
    // the await below, since the button only disables on `busy` and a slow release let several
    // clicks each resume into their own generateAudio.
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
    if (musicGeneration && !instructions) {
      updateGenerationPhase(null);
      busyRef.current = null;
      setBusy(null);
      toast.error("Add a music description for MiniMax Music 3.");
      return;
    }
    const language = audioLanguage.trim();
    const controller = new AbortController();
    generateAbort.current = controller;
    updateGenerationPhase("generating");
    try {
      const generated = await generateAudio(text, {
        ...(!musicGeneration && temperatureEdited ? { temperature } : {}),
        max_tokens: musicGeneration
          ? minimaxMusicFramesForSeconds(minimaxMaxSeconds)
          : mossFrameLimit !== null
            ? mossTtsFramesForSeconds(mossMaxSeconds, mossFrameLimit)
            : maxTokens,
        ...(instructionsKind !== null && instructions
          ? { audio_instructions: instructions }
          : {}),
        ...(mossLocalGeneration && language
          ? { audio_language: language }
          : {}),
        signal: controller.signal,
      });
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
        // The server did persist it; only this refresh missed it. Select the id so a later refresh shows
        // the real record, but keep the response audio too: selectedClip resolves against `clips`,
        // so an id that is not there yet would render the empty state.
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
    prompt,
    audioInstructions,
    audioLanguage,
    musicGeneration,
    mossLocalGeneration,
    mossFrameLimit,
    mossMaxSeconds,
    minimaxMaxSeconds,
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
      // Or a failure reads as this file's transcript, exported under its name.
      clearTranscript();
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
        const message =
          error instanceof Error ? error.message : "Transcription failed.";
        if (activeRef.current) setTranscriptError(message);
        toast.error(message);
      } finally {
        if (transcriptionAbort.current === controller) {
          transcriptionAbort.current = null;
          setBusy(null);
          if (activeRef.current) void refreshSttStatus();
        }
      }
    },
    [
      clearTranscript,
      selectedSttRepo,
      sttLoadedModel,
      sttLoadedEngine,
      refreshSttStatus,
    ],
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
      const recorder = createAudioRecorder(stream);
      // WAV is uncompressed, so on the PCM path the byte cap is reached long before the 30 minute one.
      // Express it as a duration the timer below already enforces.
      const maxSeconds =
        recorder instanceof PcmRecorder
          ? Math.min(
              RECORDING_MAX_SECONDS,
              recorder.secondsWithin(RECORDING_MAX_BYTES),
            )
          : RECORDING_MAX_SECONDS;
      const chunks: Blob[] = [];
      let recordedBytes = 0;
      let limitHit: "duration" | "size" | null = null;
      // The sidecar rejects anything past 30 minutes, and a timeslice keeps the chunks in our array
      // rather than inside the browser, so an over-long recording can be stopped at the limit.
      const stopAtLimit = (reason: "duration" | "size") => {
        if (limitHit) return;
        limitHit = reason;
        toast.warning(
          reason === "duration"
            ? `Recording stopped at the ${Math.floor(maxSeconds / 60)} minute limit.`
            : "Recording stopped: it reached the maximum upload size.",
        );
        try {
          recorder.stop();
        } catch {
          // Already stopping; the stop handler still runs.
        }
      };
      const durationTimer = window.setTimeout(
        () => stopAtLimit("duration"),
        maxSeconds * 1000,
      );
      recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size > 0) {
          // Stop before appending the chunk that crosses the limit, so what is uploaded is always inside it.
          if (recordedBytes + event.data.size > RECORDING_MAX_BYTES) {
            stopAtLimit("size");
            return;
          }
          chunks.push(event.data);
          recordedBytes += event.data.size;
        }
      });
      recorder.addEventListener("stop", () => {
        window.clearTimeout(durationTimer);
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
      // A timeslice is what makes the byte cap observable: with none, some browsers hold the whole
      // recording internally and only emit it on stop.
      recorder.start(RECORDING_CHUNK_MS);
      setIsRecording(true);
    } catch {
      // getUserMedia may have succeeded even if MediaRecorder construction failed, so release that
      // stream instead of leaving the mic live with no recorder UI to stop it.
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

  // Release the microphone on unmount AND whenever the page goes inactive: the page stays mounted
  // across tab switches, so unmount alone left a hidden recorder capturing.
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
    // Deferred like the gallery download below: browsers that resolve the synthetic navigation
    // asynchronously were left with a revoked URL and no file.
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
  }, [transcript, transcribedName]);


  const dropClip = useCallback((id: string) => {
    galleryCache.srcById.delete(id);
    setSrcById(galleryCache.srcById.toRecord());
    // Drop the row now, as the clear-all path does: refreshGallery swallows a failed GET and returns
    // the cache without setClips, leaving the row up against an already-revoked URL.
    galleryCache.clips = galleryCache.clips.filter((clip) => clip.id !== id);
    setClips(galleryCache.clips);
    if (galleryCache.selectedId === id) {
      galleryCache.selectedId = null;
      setSelectedId(null);
    }
  }, []);

  const handleDeleteClip = useCallback(
    async (id: string) => {
      try {
        await deleteAudioClip(id);
        dropClip(id);
        await refreshGallery(id);
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : "Could not delete the clip.",
        );
      }
    },
    [dropClip, refreshGallery],
  );

  const handleArchiveClip = useCallback(
    async (id: string) => {
      try {
        await setAudioClipFlags(id, { archived: true });
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : "Could not archive the clip.",
        );
        return;
      }
      dropClip(id);
      await refreshGallery(id);
      const toastId = toast(
        <button
          type="button"
          onClick={() => {
            toast.dismiss(toastId);
            useSettingsDialogStore.getState().openArchivedMedia("audio");
          }}
          className="w-full cursor-pointer text-left"
        >
          You can view archived audio in Settings
        </button>,
        { closeButton: true },
      );
    },
    [dropClip, refreshGallery],
  );

  // This page stays mounted across route changes, so a restore from the Settings archive would not
  // reach History until a reload. Refresh the loaded window, not just the first page: a clip
  // re-enters at its own age.
  useEffect(
    () =>
      subscribeGalleryChanged("audio", () => {
        void refreshGallery(undefined, galleryCache.clips.length);
      }),
    [refreshGallery],
  );

  const handleClearGallery = useCallback(async () => {
    try {
      await clearAudioGallery();
      galleryCache.srcById.clear();
      galleryCache.selectedId = null;
      // Drop the cached list first: refreshGallery merges the fetched page into it, so an empty page
      // would leave every cleared row on screen.
      galleryCache.clips = [];
      // React state too, not just the cache: refreshGallery swallows a failed GET and returns the cache
      // without calling setClips, which left cleared rows rendered against a revoked URL.
      setClips([]);
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
      // A history-row download does not need to become resident playback state, so revoke it after the
      // browser has consumed the synthetic click rather than bypassing the 64 MB cache budget.
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

  // Trained TTS checkpoints. A scan row carries no modality until the backend tags it, so without
  // this a checkpoint you just fine-tuned here is unreachable.
  const [trainedTtsModels, setTrainedTtsModels] = useState<ModelOption[]>([]);
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    listLoras()
      .then((res) => {
        if (cancelled) return;
        setTrainedTtsModels(
          res.loras
            // Merged native speech checkpoints bypass MLX through the portable audio worker. Other
            // safetensors exports still need a GGUF build on Mac.
            .filter(
              (lora) =>
                !isMac ||
                trainedTtsCheckpointIsRunnableOnMac(
                  lora.audio_type,
                  lora.export_type,
                ),
            )
            // The GGUF flag matters: GGUF_TTS_AUDIO_TYPES leaves csm out because llama.cpp has no CSM
            // decoder, so a csm LoRA exported to GGUF fails at load.
            .filter((lora) =>
              isTtsAudioType(lora.audio_type, lora.export_type === "gguf"),
            )
            .filter((lora) =>
              trainedTtsCheckpointIsLoadable(lora.audio_type, lora.export_type),
            )
            .map((lora) => ({
              id: lora.adapter_path,
              name: audioModelLabel(lora.adapter_path),
              description:
                lora.export_type === "merged"
                  ? `Fine-tuned - ${lora.base_model || "unknown base"}`
                  : `LoRA - ${lora.base_model || "unknown base"}`,
              audioType: lora.audio_type ?? null,
            })),
        );
      })
      .catch(() => {
        // Listing trained models is additive; the catalog rows still work without it.
        if (!cancelled) setTrainedTtsModels([]);
      });
    return () => {
      cancelled = true;
    };
  }, [active, isMac]);


  const selectedClip = clips.find((c) => c.id === selectedId) ?? null;
  const selectedClipSrc = selectedClip ? srcById[selectedClip.id] : undefined;
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
    <div className="@container flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden pt-[var(--studio-content-top-inset,0px)]">
      {/* Keep the tabs centered over the preview at every width. The model rail holds at 408px when
          space permits and shrinks only to preserve the controls. */}
      <div className="pointer-events-none relative z-40 grid h-[48px] shrink-0 grid-cols-[minmax(0,408px)_minmax(13rem,1fr)]">
        <div className="pointer-events-none flex h-full min-w-0 items-start overflow-hidden pl-[var(--studio-media-header-left-inset,1.5rem)] @[50rem]:border-r @[50rem]:border-border/60">
          {/* A long resident model name must yield to the mode pill instead of painting over it. */}
          <div className="pointer-events-auto flex min-w-0 max-w-full items-center gap-2 overflow-hidden pt-[var(--studio-chat-header-padding-top,11px)]">
            <ModelSelector
              models={selectorModels}
              additionalOnDeviceModels={
                mode === "transcribe" ? sttOnDeviceModels : trainedTtsModels
              }
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
              className="!h-[34px] max-w-full gap-1 overflow-hidden pl-3 pr-1 @[68rem]:gap-2 @[68rem]:pl-4 @[68rem]:pr-2"
              triggerLabelClassName="text-ui-14 @[68rem]:text-ui-16"
              task={HUB_TASKS_BY_MODE[mode]}
              catalog={AUDIO_CATALOG}
              // TTS/ASR come from the checkpoint's own tokenizer, not a curated recipe, so any publisher's
              // audio repo loads here.
              communityModelPolicy="search-only"
              placeholder="Select audio model"
              open={active && selectorOpen}
              onOpenChange={(o) => setSelectorOpen(active && o)}
            />
          </div>
        </div>
        <div className="grid h-full min-w-0 grid-cols-[1fr_auto] @[50rem]:grid-cols-[1fr_auto_1fr]">
          <div className="pointer-events-auto col-start-2 justify-self-end pr-3 pt-[var(--studio-chat-header-padding-top,11px)] @[50rem]:justify-self-center @[50rem]:pr-0">
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
              className="h-[34px] [&>button]:h-[34px] [&>button]:px-3 @[68rem]:[&>button]:px-11"
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
      </div>
      {/* Below 50rem the panes stack and the page scrolls as one column, matching Images and Video:
          side by side, the 408px rail plus a usable preview needs more width. */}
      <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-y-auto overflow-x-hidden @[50rem]:flex-row @[50rem]:overflow-hidden">
        <div className="flex w-full shrink-0 flex-col border-b border-border/60 @[50rem]:w-[408px] @[50rem]:overflow-hidden @[50rem]:border-r @[50rem]:border-b-0">
          <div
            ref={attachSettingsScroll}
            onScroll={onSettingsScroll}
            className={cn(
              "hover-scrollbar flex min-h-0 flex-1 flex-col gap-4 px-10 pt-9 pb-6 @[50rem]:overflow-y-auto",
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
                  label={musicGeneration ? "Lyrics" : "Text"}
                  htmlFor="audio-prompt"
                  hint={
                    musicGeneration
                      ? "Lyrics may use sections such as [verse] and [chorus]. The completed song lands in the gallery on the right."
                      : "What the model should say. Generation runs on the loaded TTS model and lands in the gallery on the right."
                  }
                >
                  <Textarea
                    id="audio-prompt"
                    value={prompt}
                    onChange={(event) => setPrompt(event.target.value)}
                    placeholder={
                      musicGeneration
                        ? "[verse]\nMorning light through the pines…\n\n[chorus]\n…"
                        : "Type the sentence to speak…"
                    }
                    className="min-h-28"
                  />
                </Field>
                {instructionsKind !== null ? (
                  <Field
                    label={
                      instructionsKind === "music"
                        ? "Music description"
                        : instructionsKind === "scene"
                          ? "Scene description"
                          : "Style instructions"
                    }
                    hint={
                      instructionsKind === "music"
                        ? "Describe genre, tempo, mood, vocals, and arrangement. MiniMax Music 3 requires this separately from the lyrics."
                        : instructionsKind === "scene"
                          ? "Optional Higgs TTS 2 scene guidance such as room acoustics, recording conditions, or background ambience."
                          : "Optional MOSS Local guidance such as speaking style, emotion, pace, or delivery."
                    }
                    htmlFor="audio-instructions"
                  >
                    <Textarea
                      id="audio-instructions"
                      value={audioInstructions}
                      onChange={(event) =>
                        setAudioInstructions(event.target.value)
                      }
                      placeholder={
                        instructionsKind === "music"
                          ? "Acoustic pop, 96 BPM, warm female lead, fingerpicked guitar and soft piano…"
                          : instructionsKind === "scene"
                            ? "Close-mic studio recording in a quiet, softly treated room…"
                            : "Warm, measured delivery with a calm conversational tone…"
                      }
                      className="min-h-24"
                    />
                  </Field>
                ) : null}
                {mossLocalGeneration ? (
                  <Field
                    label="Language"
                    htmlFor="audio-language"
                    hint="Optional, but MOSS Local v1.5 recommends a language tag when known (for example English, Arabic, or French)."
                  >
                    <Input
                      id="audio-language"
                      value={audioLanguage}
                      onChange={(event) => setAudioLanguage(event.target.value)}
                      placeholder="English"
                    />
                  </Field>
                ) : null}
                {/* Field inlined: its label needs a form control to point
                    at, and PillTabs is a tablist with its own name. */}
                <div className="grid gap-1.5">
                  <span className="text-ui-13 font-medium text-foreground">
                    Load model into
                  </span>
                  <PillTabs
                    ariaLabel="Load model into"
                    value={audioDevice === "cpu" ? "cpu" : "auto"}
                    // The eject below applies the change and cannot interrupt a load.
                    disabled={busy !== null || isRecording}
                    onValueChange={(value) => {
                      const next = value === "cpu" ? "cpu" : "auto";
                      if (next === audioDevice) return;
                      // MiniMax needs CUDA, and the backend's refusal cannot save a
                      // model already ejected here.
                      if (next === "cpu" && status?.audio_type === "minimax_music3") {
                        toast.info(
                          "MiniMax Music 3 needs a GPU, so it cannot be held in CPU RAM.",
                        );
                        return;
                      }
                      setAudioDeviceState(next);
                      if (ttsLoaded) handleEject();
                    }}
                    fit={true}
                    className="h-[30px] self-start [&>button]:h-[30px] [&>button]:px-6"
                    tabs={[
                      { value: "auto", label: "GPU when available" },
                      { value: "cpu", label: "CPU RAM" },
                    ]}
                  />
                  {/* Phrased as what the next load will do, not as the resident
                      model's state: a model loaded by another tab or client can
                      be on the other device, and status does not report it. */}
                  <p className="text-ui-11p5 leading-snug text-muted-foreground">
                    {audioDevice === "cpu"
                      ? "New loads go into system RAM instead of the GPU. Slower to generate, and no GPU memory is used."
                      : "New loads use the GPU when there is one, and the CPU otherwise."}
                  </p>
                </div>
                <AdvancedDisclosure
                  open={advancedOpen}
                  onOpenChange={setAdvancedOpen}
                  description={
                    musicGeneration
                      ? "Generation length. Changes apply to the next audio clip."
                      : "Generation sampling. Changes apply to the next audio clip."
                  }
                >
                  {!musicGeneration ? (
                    <ParamSlider
                      label="Temperature"
                      value={temperature}
                      min={0}
                      max={mossFrameLimit !== null ? 2 : 1.5}
                      step={0.05}
                      onChange={handleTemperatureChange}
                    />
                  ) : null}
                  {musicGeneration ? (
                    <ParamSlider
                      label="Max duration (seconds)"
                      value={minimaxMaxSeconds}
                      min={1}
                      max={MINIMAX_MUSIC_MAX_SECONDS}
                      step={1 / MINIMAX_MUSIC_FRAMES_PER_SECOND}
                      onChange={setMinimaxMaxSeconds}
                      valueSize={8}
                      info={`Starts at ${MINIMAX_MUSIC_DEFAULT_SECONDS} seconds. MiniMax Music 3 generates ${MINIMAX_MUSIC_FRAMES_PER_SECOND} frames per second, up to ${MINIMAX_MUSIC_MAX_SECONDS} seconds.`}
                    />
                  ) : mossFrameLimit !== null ? (
                    <ParamSlider
                      label="Max duration (seconds)"
                      value={mossMaxSeconds}
                      min={1}
                      max={mossMaxSecondsLimit}
                      step={1 / MOSS_TTS_FRAMES_PER_SECOND}
                      onChange={setMossMaxSeconds}
                      valueSize={8}
                      info={`Starts at ${MOSS_TTS_DEFAULT_SECONDS} seconds. This model reports ${mossFrameLimit?.toLocaleString()} frames (${mossMaxSecondsLimit.toLocaleString(undefined, { maximumFractionDigits: 2 })} seconds); the prompt uses part of that context.`}
                    />
                  ) : (
                    <ParamSlider
                      label="Max tokens"
                      value={maxTokens}
                      min={256}
                      max={TTS_MAX_TOKENS}
                      step={256}
                      onChange={setMaxTokens}
                    />
                  )}
                </AdvancedDisclosure>
              </>
            ) : (
              <>
                <Field
                  label="Microphone"
                  htmlFor="audio-record"
                  hint={
                    recordingSupported
                      ? "Record a clip and it is transcribed when you stop."
                      : "This browser cannot record. Open Unsloth over https or on localhost, or upload a file below."
                  }
                >
                  <Button
                    id="audio-record"
                    variant={isRecording ? "destructive" : "secondary"}
                    disabled={
                      !recordingSupported ||
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
                  htmlFor="audio-file"
                  hint="Or transcribe an existing recording (wav, mp3, m4a, webm…)."
                >
                  <input
                    id="audio-file"
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
            /* The scroll mask provides the fade; leave the footer unpainted to avoid dark-mode banding. */
            <div className="relative z-10 flex shrink-0 justify-center px-10 pt-0.5 pb-4">
              <div className="flex w-full max-w-sm flex-col gap-2">
                {busy === "generating" && generationPresentation ? (
                  <>
                    <output
                      aria-live="polite"
                      aria-atomic="true"
                      className="text-center text-ui-12 text-muted-foreground"
                    >
                      {generationPresentation.status}
                    </output>
                    <Progress
                      indeterminate
                      aria-label="Audio task in progress"
                      className="h-1.5"
                    />
                  </>
                ) : null}
                <Button
                  className="relative z-10 mx-auto h-11 px-8 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100"
                  onClick={
                    generationPresentation?.canStop
                      ? handleStopGeneration
                      : handleGenerate
                  }
                  disabled={
                    generationPresentation
                      ? !generationPresentation.canStop
                      : busy !== null ||
                        !ttsLoaded ||
                        !prompt.trim() ||
                        (musicGeneration && !audioInstructions.trim())
                  }
                  variant={
                    generationPresentation?.canStop ? "destructive" : "default"
                  }
                >
                  {generationPresentation?.canStop ? (
                    <>
                      <HugeiconsIcon icon={StopIcon} className="mr-2 size-4" />
                      Stop
                    </>
                  ) : (
                    (generationPresentation?.actionLabel ?? "Generate")
                  )}
                </Button>
              </div>
            </div>
          ) : null}
        </div>

        <div className="relative flex min-h-[60dvh] min-w-0 flex-1 flex-col overflow-hidden @[50rem]:min-h-0">
          {mode === "transcribe" ? (
            <div
              data-reload-snapshot-sensitive={
                transcript || transcribedName ? "" : undefined
              }
              className="hover-scrollbar flex flex-1 flex-col gap-3 overflow-auto p-6 px-10 @[50rem]:pt-[60px]"
            >
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
              ) : transcriptError ? (
                <div className="flex flex-col gap-1" role="alert">
                  <p className="text-ui-13 font-medium text-destructive">
                    Could not transcribe {transcribedName ?? "that audio"}.
                  </p>
                  <p className="text-ui-13 text-muted-foreground">
                    {transcriptError}
                  </p>
                </div>
              ) : busy !== "transcribing" ? (
                <p className="text-ui-13 text-muted-foreground">
                  The transcript appears here. It is not stored: copy or
                  download what you want to keep.
                </p>
              ) : null}
            </div>
          ) : (
            <div className="flex min-h-0 flex-1 flex-col gap-4 p-6 px-10 @[50rem]:pt-[60px]">
              <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-4">
                {selectedClip ? (
                  <div className="flex w-full max-w-xl flex-col gap-3">
                    <p className="line-clamp-2 text-ui-13 text-muted-foreground">
                      {selectedClip.prompt}
                    </p>
                    {/* Auth-protected bytes, so mount a fresh player only once this clip's object URL exists: reusing
                        one media element while src is changing left History switches showing broken controls. */}
                    {selectedClipSrc ? (
                      <audio
                        key={selectedClip.id}
                        controls={true}
                        src={selectedClipSrc}
                        className="w-full"
                      />
                    ) : (
                      <div
                        role="status"
                        className="flex h-12 w-full items-center justify-center rounded-md border border-border text-ui-12 text-muted-foreground"
                      >
                        Loading audio…
                      </div>
                    )}
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
                        {fallbackClip.model}
                        {fallbackClip.saved
                          ? " · saved, waiting for the gallery"
                          : " · not saved to the gallery"}
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
                          aria-current={
                            clip.id === selectedId ? "true" : undefined
                          }
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
                          onArchive={() => void handleArchiveClip(clip.id)}
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
