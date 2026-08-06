// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Audio page: Create (Generate = TTS via the main inference slot, Transcribe
// = STT via the dictation sidecar) and Train (LoRA on the generic trainer). The
// page stays mounted across tab switches (see __root.tsx), so `active` gates
// polling and popovers rather than lifecycle.

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
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
import { TestTubeOutlineIcon } from "@/lib/hugeicons-derived";

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
import { AdvancedDisclosure } from "@/components/advanced-disclosure";
import { useScrollFades } from "@/hooks/use-scroll-fades";
import { ModelSelector } from "@/features/model-picker/components/model-selector";
import { AUDIO_GEN_TASKS } from "@/features/model-picker/components/model-selector/pickers";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import { AUDIO_CATALOG } from "@/features/model-picker/components/model-selector/model-catalog";
import type {
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker/components/model-selector/types";
import {
  getInferenceStatus,
  loadModel,
  ParamSlider,
  type InferenceStatusResponse,
} from "@/features/chat";
import {
  SttModelNotDownloadedError,
  fetchSttStatus,
  loadSttModel,
  startSttDownload,
  sttEngineFor,
  transcribeAudioBlob,
} from "@/features/chat/adapters/studio-model-dictation-adapter";
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
import { usePlatformStore } from "@/config/env";
import { cn } from "@/lib/utils";
import { BlobUrlCache } from "@/lib/blob-url-cache";
import { toast } from "@/lib/toast";
import { useNavigate, useSearch } from "@tanstack/react-router";

import {
  clearAudioGallery,
  deleteAudioClip,
  fetchClipObjectUrl,
  generateAudio,
  listAudioGallery,
  type AudioGalleryClip,
} from "./api";
import {
  audioCapabilityLine,
  audioModelsForTask,
  audioTaskFor,
  ggufSiblingFor,
  sttSidecarKeyFor,
} from "./catalog";

const MODELS_BY_MODE: Record<CreateMode, ModelOption[]> = {
  speak: audioModelsForTask("tts"),
  transcribe: audioModelsForTask("stt"),
};

const PAGE_SIZE = 50;
// WAV clips run a few MB a minute; 64 MB keeps a healthy scrollback resident.
const CLIP_BLOB_BUDGET_BYTES = 64 * 1024 * 1024;

// Module scope so a tab switch re-renders the gallery instantly (the page stays mounted, but a
// remount after an unlikely unmount still reuses fetched clips).
const galleryCache: {
  clips: AudioGalleryClip[];
  hasMore: boolean;
  selectedId: string | null;
  srcById: BlobUrlCache;
} = {
  clips: [],
  hasMore: false,
  selectedId: null,
  srcById: new BlobUrlCache(CLIP_BLOB_BUDGET_BYTES),
};

type Busy = "loading" | "generating" | "transcribing" | null;
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
        <p className="text-ui-11p5 leading-snug text-muted-foreground">{hint}</p>
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
  const [busy, setBusy] = useState<Busy>(null);

  // --- TTS (main inference slot) -----------------------------------------
  const [status, setStatus] = useState<InferenceStatusResponse | null>(null);
  const [prompt, setPrompt] = useState("");
  const [temperature, setTemperature] = useState(0.6);
  const [maxTokens, setMaxTokens] = useState(2048);
  const generateAbort = useRef<AbortController | null>(null);

  // --- STT (dictation sidecar) -------------------------------------------
  const [selectedSttRepo, setSelectedSttRepo] = useState<string | null>(null);
  const [sttLoadedModel, setSttLoadedModel] = useState<string | null>(null);
  const [transcript, setTranscript] = useState("");
  const [transcribedName, setTranscribedName] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recordStreamRef = useRef<MediaStream | null>(null);

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
    try {
      setStatus(await getInferenceStatus());
    } catch {
      // A backend restart mid-poll is recoverable; the next refresh resyncs.
    }
  }, []);

  const refreshSttStatus = useCallback(async () => {
    try {
      const stt = await fetchSttStatus();
      setSttLoadedModel(stt.loaded_model);
    } catch {
      setSttLoadedModel(null);
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
      setClips(page.audio);
      setHasMore(page.has_more);
      if (
        galleryCache.selectedId &&
        !page.audio.some((c) => c.id === galleryCache.selectedId)
      ) {
        galleryCache.selectedId = page.audio[0]?.id ?? null;
        setSelectedId(galleryCache.selectedId);
      }
      if (!galleryCache.selectedId && page.audio.length > 0) {
        galleryCache.selectedId = page.audio[0].id;
        setSelectedId(galleryCache.selectedId);
      }
    } catch {
      // Same recoverable-poll stance as status.
    }
  }, []);

  const loadMore = useCallback(async () => {
    try {
      const page = await listAudioGallery(galleryCache.clips.length, PAGE_SIZE);
      galleryCache.clips = [...galleryCache.clips, ...page.audio];
      galleryCache.hasMore = page.has_more;
      setClips(galleryCache.clips);
      setHasMore(page.has_more);
    } catch {
      // Retry on the next scroll.
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
  }, []);

  // --- Model selection ----------------------------------------------------

  const loadTtsModel = useCallback(
    async (repoId: string, ggufFilename?: string | null) => {
      setBusy("loading");
      const toastId = toast.loading(`Loading ${repoId}…`);
      try {
        const res = await loadModel({
          model_path: repoId,
          hf_token: hfApiToken(getHfToken()) ?? null,
          max_seq_length: 2048,
          load_in_4bit: false,
          is_lora: false,
          gguf_variant: ggufFilename ?? null,
        });
        if (!res.is_audio) {
          toast.error(`${repoId} loaded but is not a TTS audio model.`, {
            id: toastId,
          });
        } else {
          toast.success(`Model loaded (${res.audio_type ?? "audio"})`, {
            id: toastId,
          });
        }
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : "Model load failed.",
          { id: toastId },
        );
      } finally {
        setBusy(null);
        void refreshStatus();
      }
    },
    [refreshStatus],
  );

  const ensureSttLoaded = useCallback(async (sidecarKey: string) => {
    setBusy("loading");
    const toastId = toast.loading(`Preparing ${sidecarKey}…`);
    try {
      try {
        await loadSttModel(sidecarKey);
      } catch (error) {
        if (!(error instanceof SttModelNotDownloadedError)) throw error;
        await startSttDownload(sidecarKey, hfApiToken(getHfToken()));
        // Poll until the sidecar reports the download finished, then load.
        for (;;) {
          await new Promise((resolve) => setTimeout(resolve, 1000));
          const stt = await fetchSttStatus(undefined, sidecarKey);
          const engine = sttEngineFor(sidecarKey);
          const block =
            engine === "mtmd"
              ? stt.mtmd
              : engine === "gguf"
                ? stt.gguf
                : stt.transformers;
          const download = block?.download;
          if (download?.error) throw new Error(download.error);
          if (!download?.downloading) break;
          const total = download.bytes_total ?? 0;
          const done = download.bytes_done ?? 0;
          if (total > 0) {
            toast.loading(
              `Downloading ${sidecarKey}: ${Math.round((done / total) * 100)}%`,
              { id: toastId },
            );
          }
        }
        await loadSttModel(sidecarKey);
      }
      toast.success("Transcription model ready", { id: toastId });
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Transcription model failed.",
        { id: toastId },
      );
    } finally {
      setBusy(null);
      void refreshSttStatus();
    }
  }, [refreshSttStatus]);

  const isMac = usePlatformStore((s) => s.deviceType) === "mac";

  const handleModelSelect = useCallback(
    (id: string, meta: ModelSelectorChangeMeta) => {
      const task = audioTaskFor(id);
      if (task === "stt") {
        // An STT pick owns Transcribe: it runs on the sidecar, not the main slot.
        setMode("transcribe");
        setSelectedSttRepo(id);
        void ensureSttLoaded(sttSidecarKeyFor(id));
        return;
      }
      // TTS (or an uncurated repo the user pasted, which /load will validate).
      setMode("speak");
      // Safetensors here loads on MLX and then 501s on every generation, so take
      // the GGUF build (see ggufSiblingFor) and say so.
      const gguf = isMac && !meta.ggufFilename ? ggufSiblingFor(id) : null;
      if (gguf) {
        toast.info(
          `Loading the GGUF build of ${id}. MLX has no text-to-speech decoder, so the safetensors build cannot generate on this Mac.`,
          { duration: 7000 },
        );
        void loadTtsModel(gguf, null);
        return;
      }
      void loadTtsModel(id, meta.ggufFilename ?? null);
    },
    [ensureSttLoaded, loadTtsModel, isMac],
  );

  // A pick handed over from the chat model selector arrives as ?model= (+ ?quant=).
  const navigateSelf = useNavigate();
  const routeSearch = useSearch({ strict: false }) as {
    model?: string;
    quant?: string;
  };
  const handledRouteModel = useRef<string | null>(null);
  useEffect(() => {
    if (!active) return;
    const wanted = routeSearch.model;
    if (!wanted) {
      handledRouteModel.current = null;
      return;
    }
    const key = `${wanted}|${routeSearch.quant ?? ""}`;
    if (handledRouteModel.current === key) return;
    handledRouteModel.current = key;
    void navigateSelf({ to: "/audio", search: {}, replace: true });
    handleModelSelect(wanted, {
      source: "hub",
      isLora: false,
      ggufFilename: routeSearch.quant ?? undefined,
    });
  }, [active, routeSearch.model, routeSearch.quant, handleModelSelect, navigateSelf]);

  // --- Speak --------------------------------------------------------------

  const ttsLoaded = Boolean(status?.is_audio && status?.active_model);
  const handleGenerate = useCallback(async () => {
    const text = prompt.trim();
    if (!text) return;
    setBusy("generating");
    const controller = new AbortController();
    generateAbort.current = controller;
    try {
      await generateAudio(text, {
        temperature,
        max_tokens: maxTokens,
        signal: controller.signal,
      });
      await refreshGallery();
      const newest = galleryCache.clips[0];
      if (newest) selectClip(newest.id);
    } catch (error) {
      if (!controller.signal.aborted) {
        toast.error(
          error instanceof Error ? error.message : "Audio generation failed.",
        );
      }
    } finally {
      generateAbort.current = null;
      setBusy(null);
    }
  }, [prompt, temperature, maxTokens, refreshGallery, selectClip]);

  // Abort an in-flight generation on a true unmount; the backend cancels on disconnect.
  useEffect(() => () => generateAbort.current?.abort(), []);

  // --- Transcribe ---------------------------------------------------------

  const sttSelected = selectedSttRepo !== null;
  const runTranscription = useCallback(
    async (blob: Blob, name: string) => {
      if (!selectedSttRepo) return;
      setBusy("transcribing");
      setTranscribedName(name);
      try {
        const key = sttSidecarKeyFor(selectedSttRepo);
        const text = await transcribeAudioBlob(blob, {
          model: key,
          engine: sttEngineFor(key),
        });
        setTranscript(text);
        if (!text) toast.info("The model heard no speech in that audio.");
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : "Transcription failed.",
        );
      } finally {
        setBusy(null);
        void refreshSttStatus();
      }
    },
    [selectedSttRepo, refreshSttStatus],
  );

  const stopRecordStream = useCallback(() => {
    for (const track of recordStreamRef.current?.getTracks() ?? []) track.stop();
    recordStreamRef.current = null;
  }, []);

  const handleRecordToggle = useCallback(async () => {
    if (isRecording) {
      recorderRef.current?.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      recordStreamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size > 0) chunks.push(event.data);
      });
      recorder.addEventListener("stop", () => {
        setIsRecording(false);
        stopRecordStream();
        recorderRef.current = null;
        const blob = new Blob(chunks, {
          type: recorder.mimeType || "audio/webm",
        });
        if (blob.size > 0) void runTranscription(blob, "Recording");
      });
      recorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
    } catch {
      toast.error("Could not access the microphone.");
    }
  }, [isRecording, runTranscription, stopRecordStream]);

  // Release the microphone if the page unmounts mid-recording.
  useEffect(
    () => () => {
      recorderRef.current?.stop();
      stopRecordStream();
    },
    [stopRecordStream],
  );

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

  /** Download from a history row, whose bytes are only fetched once selected. */
  const handleDownloadClipById = useCallback(async (clip: AudioGalleryClip) => {
    try {
      let src = galleryCache.srcById.get(clip.id);
      if (!src) {
        const fetched = await fetchClipObjectUrl(clip.url);
        galleryCache.srcById.set(clip.id, fetched.url, fetched.bytes);
        setSrcById(galleryCache.srcById.toRecord());
        src = fetched.url;
      }
      const anchor = document.createElement("a");
      anchor.href = src;
      anchor.download = `${clip.id}.wav`;
      anchor.click();
    } catch {
      toast.error("Could not download the clip.");
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
  const selectorValue =
    mode === "speak"
      ? ttsLoaded
        ? (status?.active_model ?? undefined)
        : undefined
      : (selectedSttRepo ?? undefined);

  const capabilityLine =
    mode === "speak"
      ? ttsLoaded
        ? audioCapabilityLine("tts", status?.audio_type)
        : status?.active_model
          ? "The loaded model is not a TTS audio model."
          : "No TTS model loaded."
      : sttSelected
        ? audioCapabilityLine(
            "stt",
            sttLoadedModel === sttSidecarKeyFor(selectedSttRepo ?? "")
              ? "ready"
              : "loads on first use",
          )
        : "No transcription model selected.";

  return (
    <div className="flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden pt-[var(--studio-content-top-inset,0px)]">
      <div className="relative flex h-[48px] shrink-0 items-start justify-between pl-[var(--studio-media-header-left-inset,1.5rem)] pr-2 pt-[var(--studio-chat-header-padding-top,11px)]">
        <div className="flex items-center gap-2">
          <ModelSelector
            models={MODELS_BY_MODE[mode]}
            value={selectorValue}
            onValueChange={handleModelSelect}
            variant="ghost"
            className="!h-[34px]"
            task={AUDIO_GEN_TASKS}
            catalog={AUDIO_CATALOG}
            // TTS/ASR come from the checkpoint's own tokenizer, not a curated
            // recipe, so any publisher's audio repo loads here.
            includeCommunity={true}
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
                icon: <HugeiconsIcon icon={SparklesIcon} className="size-3.5" />,
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
                onValueChange={(v) => setMode(v as CreateMode)}
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
                  <AdvancedDisclosure open={advancedOpen} onOpenChange={setAdvancedOpen}>
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
                      disabled={!sttSelected || busy === "transcribing"}
                      onClick={handleRecordToggle}
                    >
                      <HugeiconsIcon
                        icon={isRecording ? StopIcon : Mic01Icon}
                        className="mr-2 size-4"
                      />
                      {isRecording ? "Stop recording" : "Record"}
                    </Button>
                  </Field>
                  <Field
                    label="Audio file"
                    hint="Or transcribe an existing recording (wav, mp3, m4a, webm…)."
                  >
                    <input
                      type="file"
                      accept="audio/*"
                      disabled={!sttSelected || busy === "transcribing"}
                      onChange={(event) => {
                        handleTranscribeFile(event.target.files?.[0]);
                        event.target.value = "";
                      }}
                      className="text-ui-13 file:mr-3 file:rounded-md file:border-0 file:bg-muted file:px-3 file:py-1.5 file:text-ui-13 file:font-medium"
                    />
                  </Field>
                  {!sttSelected ? (
                    <p className="text-ui-11p5 leading-snug text-muted-foreground">
                      Pick a speech-to-text model (Whisper or Qwen3-ASR) from the
                      selector above to transcribe.
                    </p>
                  ) : null}
                </>
              )}
            </div>
            {mode === "speak" ? (
              <div className="pointer-events-none absolute inset-x-0 bottom-0 flex justify-center pb-7 pl-8 pr-8">
                <Button
                  className="btn-float-action pointer-events-auto h-11 px-8 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100"
                  onClick={handleGenerate}
                  disabled={busy !== null || !ttsLoaded || !prompt.trim()}
                >
                  {busy === "generating" ? (
                    <Spinner className="mr-2 size-4" />
                  ) : null}
                  Generate
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
                      <Button variant="secondary" size="sm" onClick={handleCopyTranscript}>
                        Copy
                      </Button>
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={handleDownloadTranscript}
                      >
                        <HugeiconsIcon icon={Download01Icon} className="mr-2 size-3.5" />
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
                          disabled={!srcById[selectedClip.id]}
                          onClick={() => handleDownloadClip(selectedClip)}
                        >
                          <HugeiconsIcon icon={Download01Icon} className="size-3.5" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => void handleDeleteClip(selectedClip.id)}
                        >
                          <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
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
                      <Button variant="ghost" size="sm" onClick={() => void handleClearGallery()}>
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
                            onCopyPrompt={() => void handleCopyPrompt(clip.prompt)}
                            onUseAsText={() => {
                              setMode("speak");
                              setPrompt(clip.prompt);
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
