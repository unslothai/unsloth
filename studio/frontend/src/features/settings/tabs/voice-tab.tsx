// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import {
  StudioModelDictationAdapter,
  type SttDownloadStatus,
  fetchSttStatus,
  loadSttModel,
  startSttDownload,
  unloadSttModel,
  validateSttModel,
} from "@/features/chat/adapters/studio-model-dictation-adapter";
import {
  StudioSpeechSynthesisAdapter,
  createConfiguredUtterance,
  curateSystemVoices,
  generateStudioTtsAudio,
} from "@/features/chat/adapters/studio-speech-synthesis-adapter";
import { DownloadProgressBar } from "@/features/hub";
import { useHubModelSearch } from "@/features/hub/hooks/use-hub-model-search";
import {
  hfApiToken,
  useHfTokenStore,
} from "@/features/hub/stores/hf-token-store";
import { useDebouncedValue } from "@/hooks";
import { useT } from "@/i18n";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { MicIcon } from "@/lib/mic-icon";
import { toast } from "@/lib/toast";
import { Search01Icon, VolumeHighIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { SquareIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DictationDictionaryView } from "../components/dictation-dictionary-view";
import { RecentDictationsView } from "../components/recent-dictations-view";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import {
  type DefaultSttModel,
  STT_MODELS,
  type SttModel,
  getSttModelRepo,
  isCuratedSttModel,
  isSttModelId,
  isSttModelLanguageCompatible,
  useVoiceSettingsStore,
} from "../stores/voice-settings-store";

// Languages shared by browser speech recognition and local STT.
const DICTATION_LANGUAGES: { value: string; label: string }[] = [
  { value: "auto", label: "" }, // label rendered via i18n
  { value: "en-US", label: "English (US)" },
  { value: "en-GB", label: "English (UK)" },
  { value: "zh-CN", label: "中文 (简体)" },
  { value: "ja-JP", label: "日本語" },
  { value: "ko-KR", label: "한국어" },
  { value: "es-ES", label: "Español" },
  { value: "fr-FR", label: "Français" },
  { value: "de-DE", label: "Deutsch" },
  { value: "it-IT", label: "Italiano" },
  { value: "pt-BR", label: "Português (Brasil)" },
  { value: "ru-RU", label: "Русский" },
  { value: "hi-IN", label: "हिन्दी" },
  { value: "ar-SA", label: "العربية" },
];

// Speech-recognition models, not voices. Name and download size are kept apart
// so the list can right-align the size; the speed/accuracy note lives in the
// row description.
const STT_MODEL_NAMES: Record<DefaultSttModel, string> = {
  tiny: "Whisper Tiny",
  base: "Whisper Base",
  small: "Whisper Small",
  "large-v3-turbo": "Whisper Large v3 Turbo",
  "large-v3": "Whisper Large v3",
};
// Curated models run as f16 GGML files through whisper.cpp.
const STT_MODEL_SIZES: Record<DefaultSttModel, string> = {
  tiny: "78 MB",
  base: "148 MB",
  small: "488 MB",
  "large-v3-turbo": "1.6 GB",
  "large-v3": "3.1 GB",
};

function sttModelName(model: SttModel): string {
  return STT_MODEL_NAMES[model as DefaultSttModel] ?? model;
}

function sttModelSize(model: SttModel): string {
  return STT_MODEL_SIZES[model as DefaultSttModel] ?? "";
}

/** Source repository shown under a model row. Curated models download from
 * the Unsloth GGUF repos, mirrored by the backend (stt_ggml_sidecar.py). */
function sttModelSource(model: SttModel): string {
  return isCuratedSttModel(model)
    ? `unslothai/whisper-${model}-GGUF`
    : getSttModelRepo(model);
}

/**
 * Model picker for local transcription. Lists the curated whisper.cpp
 * checkpoints and searches Hugging Face for other Whisper repositories,
 * which run as safetensors through Transformers. The trigger is a plain
 * button so the selection never renders inside a text input.
 */
function SttModelPicker({
  value,
  language,
  onChange,
}: {
  value: SttModel;
  language: string;
  onChange: (model: SttModel) => void;
}) {
  const t = useT();
  const hfToken = useHfTokenStore((state) => state.token);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [validating, setValidating] = useState(false);
  const debouncedQuery = useDebouncedValue(query.trim());

  const { results, isLoading } = useHubModelSearch(debouncedQuery, {
    task: "automatic-speech-recognition",
    accessToken: hfApiToken(hfToken),
    excludeGguf: true,
    enabled: debouncedQuery.length >= 2,
    keepUnsupportedTags: true,
    ownerScope: "all",
  });

  const items = useMemo(() => {
    if (!debouncedQuery) {
      const defaults: string[] = STT_MODELS.filter((model) =>
        isSttModelLanguageCompatible(model, language),
      );
      if (!defaults.includes(value)) {
        defaults.push(value);
      }
      return defaults;
    }
    const ids: string[] = [];
    for (const result of results) {
      const tags = result.tags?.map((tag) => tag.toLowerCase()) ?? [];
      const isWhisper =
        result.id.toLowerCase().includes("whisper") || tags.includes("whisper");
      const isExactMatch =
        result.id.toLowerCase() === debouncedQuery.toLowerCase();
      if (
        (isExactMatch ||
          (isWhisper &&
            result.pipelineTag === "automatic-speech-recognition")) &&
        isSttModelLanguageCompatible(result.id, language) &&
        !ids.includes(result.id)
      ) {
        ids.push(result.id);
      }
    }
    return ids;
  }, [debouncedQuery, language, results, value]);

  const selectModel = async (model: string) => {
    if (!isSttModelId(model) || validating) {
      return;
    }
    if (!isCuratedSttModel(model)) {
      setValidating(true);
      try {
        await validateSttModel(model, hfApiToken(hfToken));
      } catch (error) {
        toast.error(t("settings.voice.dictation.sttModelInvalid"), {
          description: error instanceof Error ? error.message : undefined,
        });
        return;
      } finally {
        setValidating(false);
      }
    }
    onChange(model);
    setOpen(false);
    setQuery("");
  };

  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (!next) setQuery("");
      }}
    >
      <PopoverTrigger asChild={true}>
        <button
          type="button"
          aria-label="Speech recognition model"
          className="border-border bg-background hover:bg-accent/50 dark:border-transparent dark:bg-white/[0.06] dark:hover:bg-white/10 focus-visible:border-ring flex h-8 w-full cursor-pointer items-center justify-between gap-1.5 rounded-full border px-3.5 text-sm outline-none transition-colors"
        >
          <span className="truncate">{sttModelName(value)}</span>
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            strokeWidth={2}
            className="text-muted-foreground pointer-events-none size-4 shrink-0"
          />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" sideOffset={4} className="w-72 gap-0 p-0">
        <div className="relative p-1.5 pb-0.5">
          <HugeiconsIcon
            icon={Search01Icon}
            strokeWidth={2}
            className="text-muted-foreground pointer-events-none absolute top-[calc(50%+2px)] left-4 size-3.5 -translate-y-1/2"
          />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={t("settings.voice.dictation.sttModelSearchPlaceholder")}
            className="h-8 pl-8 text-sm"
            autoFocus={true}
            onKeyDown={(event) => {
              if (event.key === "Enter" && items.length > 0) {
                event.preventDefault();
                void selectModel(items[0]);
              }
            }}
          />
        </div>
        <div className="max-h-64 overflow-y-auto p-1">
          {(isLoading && debouncedQuery) || validating ? (
            <div className="flex items-center gap-2 px-3 py-3 text-xs text-muted-foreground">
              <Spinner className="size-3.5" />
              {validating
                ? t("settings.voice.dictation.sttModelValidating")
                : t("settings.voice.dictation.sttModelSearching")}
            </div>
          ) : items.length === 0 ? (
            <div className="px-3 py-3 text-xs text-muted-foreground">
              {t("settings.voice.dictation.sttModelNoResults")}
            </div>
          ) : (
            items.map((model) => {
              // A custom repo's name is its id; those rows are one line and
              // keep a pill shape, two-line rows use a subtler radius.
              const twoLines = sttModelSource(model) !== sttModelName(model);
              return (
                <button
                  key={model}
                  type="button"
                  onClick={() => void selectModel(model)}
                  aria-selected={model === value}
                  className={`flex w-full items-center justify-between gap-3 px-2.5 py-1.5 text-left transition-colors hover:bg-muted ${
                    twoLines ? "rounded-sm" : "rounded-full"
                  } ${model === value ? "bg-accent font-medium" : ""}`}
                >
                  <span className="min-w-0 flex-1 truncate">
                    <span className="block truncate text-xs">
                      {sttModelName(model)}
                    </span>
                    {twoLines ? (
                      <span className="mt-0.5 block truncate font-mono text-[9px] leading-tight text-muted-foreground">
                        {sttModelSource(model)}
                      </span>
                    ) : null}
                  </span>
                  {sttModelSize(model) ? (
                    <span className="shrink-0 text-[10px] tabular-nums text-muted-foreground">
                      {sttModelSize(model)}
                    </span>
                  ) : null}
                </button>
              );
            })
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}

const TTS_PREVIEW_TEXT =
  "Hello from Unsloth Studio! This is a preview of the selected voice.";

function useAudioInputDevices() {
  const t = useT();
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [hasLabels, setHasLabels] = useState(false);

  const refresh = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      const inputs = all.filter((d) => d.kind === "audioinput");
      setDevices(inputs);
      setHasLabels(inputs.some((d) => d.label));
    } catch {
      // Enumeration can fail in insecure contexts; leave the list empty.
    }
  }, []);

  useEffect(() => {
    void refresh();
    const media = navigator.mediaDevices;
    if (!media?.addEventListener) return;
    media.addEventListener("devicechange", refresh);
    return () => media.removeEventListener("devicechange", refresh);
  }, [refresh]);

  // Labels are hidden until mic permission; open a short stream to get them.
  const requestAccess = useCallback(async () => {
    // Insecure contexts (plain http on a LAN address) have no mediaDevices.
    if (!navigator.mediaDevices?.getUserMedia) {
      toast.error(t("settings.voice.dictation.micAccessUnsupported"));
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });
      for (const track of stream.getTracks()) {
        track.stop();
      }
      await refresh();
    } catch {
      toast.error(t("settings.voice.dictation.micAccessBlocked"));
    }
  }, [refresh, t]);

  return { devices, hasLabels, requestAccess };
}

function useSystemVoices() {
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);

  useEffect(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;
    const synth = window.speechSynthesis;
    const load = () => setVoices(synth.getVoices());
    load();
    synth.addEventListener?.("voiceschanged", load);
    return () => synth.removeEventListener?.("voiceschanged", load);
  }, []);

  return voices;
}

export function VoiceTab() {
  const t = useT();
  const micDeviceId = useVoiceSettingsStore((s) => s.micDeviceId);
  const setMicDeviceId = useVoiceSettingsStore((s) => s.setMicDeviceId);
  const dictationEngine = useVoiceSettingsStore((s) => s.dictationEngine);
  const setDictationEngine = useVoiceSettingsStore((s) => s.setDictationEngine);
  const sttModel = useVoiceSettingsStore((s) => s.sttModel);
  const setSttModel = useVoiceSettingsStore((s) => s.setSttModel);
  const dictationLanguage = useVoiceSettingsStore((s) => s.dictationLanguage);
  const setDictationLanguage = useVoiceSettingsStore(
    (s) => s.setDictationLanguage,
  );
  const ttsEnabled = useVoiceSettingsStore((s) => s.ttsEnabled);
  const setTtsEnabled = useVoiceSettingsStore((s) => s.setTtsEnabled);
  const ttsEngine = useVoiceSettingsStore((s) => s.ttsEngine);
  const setTtsEngine = useVoiceSettingsStore((s) => s.setTtsEngine);
  const ttsVoiceURI = useVoiceSettingsStore((s) => s.ttsVoiceURI);
  const setTtsVoiceURI = useVoiceSettingsStore((s) => s.setTtsVoiceURI);
  const ttsRate = useVoiceSettingsStore((s) => s.ttsRate);
  const setTtsRate = useVoiceSettingsStore((s) => s.setTtsRate);
  const ttsPitch = useVoiceSettingsStore((s) => s.ttsPitch);
  const setTtsPitch = useVoiceSettingsStore((s) => s.setTtsPitch);
  const ttsVolume = useVoiceSettingsStore((s) => s.ttsVolume);
  const setTtsVolume = useVoiceSettingsStore((s) => s.setTtsVolume);

  const { devices, hasLabels, requestAccess } = useAudioInputDevices();
  const rawVoices = useSystemVoices();
  const voices = useMemo(
    () => curateSystemVoices(rawVoices, ttsVoiceURI, dictationLanguage),
    [rawVoices, ttsVoiceURI, dictationLanguage],
  );
  const [previewing, setPreviewing] = useState(false);
  const [subpage, setSubpage] = useState<"main" | "recents" | "dictionary">(
    "main",
  );
  const [selectedDictationId, setSelectedDictationId] = useState<string | null>(
    null,
  );

  const modelSttSupported = StudioModelDictationAdapter.isSupported();
  const ttsSupported = StudioSpeechSynthesisAdapter.isSupported();
  const systemTtsSupported =
    StudioSpeechSynthesisAdapter.systemVoicesSupported();
  const effectiveTtsEngine = systemTtsSupported ? ttsEngine : "studio";

  // Local STT stays on-demand. Track its phase without fetching model weights.
  type SttPhase =
    | "idle"
    | "checking"
    | "on-demand"
    | "unavailable"
    | "loading"
    | "ready"
    | "error";
  type SttDownloadAvailability =
    | "checking"
    | "missing"
    | "downloaded"
    | "error";
  const [sttPhase, setSttPhase] = useState<SttPhase>("idle");
  const [sttDevice, setSttDevice] = useState<string | null>(null);
  const [statusNonce, setStatusNonce] = useState(0);
  const [sttDownloadStarting, setSttDownloadStarting] = useState(false);
  const [sttUnloading, setSttUnloading] = useState(false);
  const isLocalEngine = dictationEngine === "model";
  // The selected model decides the backend: curated ids run GGML files
  // through whisper.cpp, custom repositories run through Transformers.
  const isGgufModel = isCuratedSttModel(sttModel);
  // Progress of the selected engine's model download, from /stt/status.
  const [sttDownload, setSttDownload] = useState<SttDownloadStatus | null>(
    null,
  );
  const [downloadBytesPerSec, setDownloadBytesPerSec] = useState(0);
  // Last observed (bytes, time) so successive polls yield a transfer rate.
  const downloadRateSampleRef = useRef<{ bytes: number; at: number } | null>(
    null,
  );
  // Model whose download this tab watched; completion auto-loads it.
  const watchedDownloadRef = useRef<string | null>(null);

  // Selecting a model (or finishing its download) loads it without a Load
  // click. A model that is not downloaded fails quietly and stays on demand.
  const autoLoadSttModel = useCallback(async (model: string) => {
    setSttPhase("loading");
    try {
      await loadSttModel(model);
    } catch {
      // Not downloaded (or the engine is busy): the status poll resets the
      // phase; the user still sees the Download button.
    } finally {
      setStatusNonce((nonce) => nonce + 1);
    }
  }, []);
  const sttRepoId = getSttModelRepo(sttModel);
  const hfToken = useHfTokenStore((state) => state.token);
  const [sttDownloadAvailability, setSttDownloadAvailability] = useState<{
    repoId: string;
    state: SttDownloadAvailability;
  }>({ repoId: "", state: "checking" });
  const effectiveSttDownloadAvailability =
    sttDownloadAvailability.repoId === sttRepoId
      ? sttDownloadAvailability.state
      : "checking";
  useEffect(() => {
    if (!isLocalEngine || !modelSttSupported) {
      setSttPhase("idle");
      setSttDevice(null);
      setSttDownload(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      // Only surface "checking" on the first poll; background refreshes keep
      // the last phase so the status line doesn't flicker while polling.
      setSttPhase((phase) => (phase === "idle" ? "checking" : phase));
      try {
        const status = await fetchSttStatus(statusNonce, sttModel);
        if (cancelled) return;
        // A curated model prefers the GGUF (whisper.cpp) engine, but when
        // whisper-server is not installed the backend serves it through the
        // Transformers engine instead of failing, so fall back to the
        // Transformers status here too; otherwise the model shows as unavailable
        // and download is blocked even though dictation works.
        const engineStatus =
          isGgufModel && status.gguf?.available ? status.gguf : status.transformers;
        if (!engineStatus?.available) {
          setSttPhase("unavailable");
          return;
        }
        const download = engineStatus.download;
        setSttDownload(download);
        setSttDownloadAvailability({
          repoId: sttRepoId,
          state: engineStatus.downloaded_models.includes(sttModel)
            ? "downloaded"
            : download.error
              ? "error"
              : "missing",
        });
        if (download.downloading) {
          watchedDownloadRef.current = download.model;
          const bytes = download.bytes_done ?? 0;
          const sample = downloadRateSampleRef.current;
          const now = Date.now();
          if (sample && bytes > sample.bytes && now > sample.at) {
            setDownloadBytesPerSec(
              ((bytes - sample.bytes) * 1000) / (now - sample.at),
            );
          }
          downloadRateSampleRef.current = { bytes, at: now };
          // Keep the download progress fresh.
          window.setTimeout(() => {
            if (!cancelled) setStatusNonce((n) => n + 1);
          }, 800);
        } else {
          const finished = watchedDownloadRef.current;
          watchedDownloadRef.current = null;
          downloadRateSampleRef.current = null;
          setDownloadBytesPerSec(0);
          if (
            finished === sttModel &&
            engineStatus.downloaded_models.includes(sttModel) &&
            engineStatus.loaded_model !== sttModel
          ) {
            // The download this tab watched just finished; load the model.
            void autoLoadSttModel(sttModel);
            return;
          }
        }
        if (engineStatus.loading) {
          setSttPhase("loading");
          window.setTimeout(() => {
            if (!cancelled) setStatusNonce((n) => n + 1);
          }, 600);
          return;
        }
        if (engineStatus.loaded_model === sttModel && !engineStatus.loading) {
          setSttDevice(engineStatus.device);
          setSttPhase("ready");
          window.setTimeout(
            () => {
              if (!cancelled) setStatusNonce((n) => n + 1);
            },
            Math.max(
              1000,
              Math.min(engineStatus.keep_alive_seconds * 1000, 15_000),
            ),
          );
          return;
        }
        // Merely opening settings or selecting local STT never downloads or
        // loads a model. Loading begins from Load or when recording starts.
        setSttDevice(null);
        setSttPhase("on-demand");
      } catch {
        if (!cancelled) setSttPhase("error");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [
    isLocalEngine,
    isGgufModel,
    sttModel,
    sttRepoId,
    modelSttSupported,
    statusNonce,
    autoLoadSttModel,
  ]);

  const sttStatusText = (() => {
    switch (sttPhase) {
      case "checking":
        return t("settings.voice.dictation.sttChecking");
      case "loading":
        return t("settings.voice.dictation.sttLoadingModel");
      case "on-demand":
        return t("settings.voice.dictation.sttOnDemand");
      case "ready":
        // The GGML backend reports its runtime name, not a device; show a
        // plain "Loaded" rather than surfacing it.
        return sttDevice && sttDevice !== "whisper.cpp"
          ? t("settings.voice.dictation.sttReady", {
              device: sttDevice.toUpperCase(),
            })
          : t("settings.voice.dictation.sttLoaded");
      case "unavailable":
        return t("settings.voice.dictation.sttUnavailable");
      case "error":
        return t("settings.voice.dictation.sttModelFailed");
      default:
        return "";
    }
  })();

  const downloadingThisModel =
    isLocalEngine &&
    sttDownload?.downloading === true &&
    sttDownload.model === sttModel;

  const sttModelStatusText = (() => {
    if (downloadingThisModel) {
      const total = sttDownload?.bytes_total ?? 0;
      const done = sttDownload?.bytes_done ?? 0;
      return t("settings.voice.dictation.sttDownloading", {
        progress: total > 0 ? Math.min(99, Math.round((done / total) * 100)) : 0,
      });
    }
    if (sttPhase === "unavailable") {
      return sttStatusText;
    }
    switch (effectiveSttDownloadAvailability) {
      case "checking":
        return t("settings.voice.dictation.sttDownloadChecking");
      case "missing":
        return t("settings.voice.dictation.sttNotDownloaded");
      case "error":
        return t("settings.voice.dictation.sttDownloadStatusFailed");
      case "downloaded":
        return sttStatusText;
      default:
        return "";
    }
  })();

  const beginSttDownload = async () => {
    setSttDownloadStarting(true);
    try {
      // Engine-managed download; progress arrives via /stt/status.
      await startSttDownload(sttModel, hfApiToken(hfToken));
      setStatusNonce((nonce) => nonce + 1);
    } catch (error) {
      toast.error(t("settings.voice.dictation.sttDownloadFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setSttDownloadStarting(false);
    }
  };

  const warmSttModel = async () => {
    setSttPhase("loading");
    try {
      await loadSttModel(sttModel);
      setStatusNonce((nonce) => nonce + 1);
    } catch (error) {
      setSttPhase("error");
      toast.error(t("settings.voice.dictation.sttModelFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    }
  };


  const releaseSttModel = async () => {
    setSttUnloading(true);
    try {
      await unloadSttModel();
      setStatusNonce((nonce) => nonce + 1);
    } catch (error) {
      toast.error(t("settings.voice.dictation.sttModelFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setSttUnloading(false);
    }
  };

  // Keep an item for an unplugged saved mic so the value stays visible.
  const knownMic = devices.some((d) => d.deviceId === micDeviceId);

  const previewAudioRef = useRef<HTMLAudioElement | null>(null);
  const previewAbortRef = useRef<AbortController | null>(null);
  // Mirrors `previewing` so unmount cleanup can tell whether this tab owns
  // the current speechSynthesis utterance; read-aloud shares the global
  // synthesizer and must not be cancelled by merely closing settings.
  const previewingRef = useRef(false);
  // Only a system-voice preview owns the shared speechSynthesis channel; a
  // studio (Audio) preview must not cancel an unrelated chat read-aloud.
  const ownsSystemPreviewRef = useRef(false);
  const markPreviewing = useCallback((value: boolean) => {
    previewingRef.current = value;
    setPreviewing(value);
  }, []);

  const releasePreviewAudio = useCallback(() => {
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
      previewAudioRef.current.src = "";
      previewAudioRef.current = null;
    }
  }, []);

  const stopPreview = useCallback(() => {
    if (!previewingRef.current) return;
    if (ownsSystemPreviewRef.current) {
      window.speechSynthesis?.cancel();
      ownsSystemPreviewRef.current = false;
    }
    previewAbortRef.current?.abort();
    previewAbortRef.current = null;
    releasePreviewAudio();
    markPreviewing(false);
  }, [markPreviewing, releasePreviewAudio]);

  const previewTts = async () => {
    if (!ttsSupported) return;
    // Ref, not state: a double-click before rerender still reads previewing
    // as false and would start a second request that orphans the first.
    if (previewingRef.current) {
      stopPreview();
      return;
    }
    if (effectiveTtsEngine === "studio") {
      const controller = new AbortController();
      previewAbortRef.current = controller;
      ownsSystemPreviewRef.current = false;
      markPreviewing(true);
      try {
        const url = await generateStudioTtsAudio(
          TTS_PREVIEW_TEXT,
          controller.signal,
        );
        if (controller.signal.aborted) return;
        const audio = new Audio(url);
        audio.playbackRate = ttsRate;
        audio.volume = ttsVolume;
        // Some browsers reset playbackRate once metadata loads.
        audio.addEventListener("loadedmetadata", () => {
          audio.playbackRate = ttsRate;
        });
        audio.addEventListener("ended", () => {
          releasePreviewAudio();
          markPreviewing(false);
        });
        audio.addEventListener("error", () => {
          releasePreviewAudio();
          markPreviewing(false);
          toast.error("TTS preview failed");
        });
        previewAudioRef.current = audio;
        await audio.play();
      } catch (error) {
        if (!controller.signal.aborted) {
          toast.error(
            error instanceof Error ? error.message : "TTS preview failed",
          );
        }
        releasePreviewAudio();
        markPreviewing(false);
      }
      return;
    }
    if (!StudioSpeechSynthesisAdapter.systemVoicesSupported()) {
      toast.error(t("settings.voice.readAloud.notSupported"));
      return;
    }
    const utterance = createConfiguredUtterance(TTS_PREVIEW_TEXT);
    utterance.addEventListener("end", () => {
      ownsSystemPreviewRef.current = false;
      markPreviewing(false);
    });
    utterance.addEventListener("error", () => {
      ownsSystemPreviewRef.current = false;
      markPreviewing(false);
    });
    ownsSystemPreviewRef.current = true;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
    markPreviewing(true);
  };

  // Stop any preview playback when the tab unmounts.
  useEffect(() => stopPreview, [stopPreview]);

  if (subpage === "recents") {
    return (
      <RecentDictationsView
        selectedId={selectedDictationId}
        onSelect={setSelectedDictationId}
        onBack={() => {
          setSelectedDictationId(null);
          setSubpage("main");
        }}
      />
    );
  }

  if (subpage === "dictionary") {
    return <DictationDictionaryView onBack={() => setSubpage("main")} />;
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-xl font-semibold font-heading">
          {t("settings.voice.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.voice.description")}
        </p>
      </header>

      <SettingsSection title={t("settings.voice.dictation.sectionTitle")}>
        <SettingsRow
          label={t("settings.voice.dictation.engineLabel")}
          description={
            dictationEngine === "model"
              ? t("settings.voice.dictation.engineModelDescription")
              : t("settings.voice.dictation.engineBrowserDescription")
          }
        >
          <Select
            value={dictationEngine}
            onValueChange={(value) => {
              const next = value === "model" ? "model" : "browser";
              if (next !== dictationEngine) {
                // Unload whichever backend was resident for the old engine.
                void unloadSttModel().catch(() => {});
              }
              setDictationEngine(next);
            }}
          >
            <SelectTrigger
              aria-label="Dictation engine"
              className="w-56"
              size="sm"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="browser">
                {t("settings.voice.dictation.engineBrowser")}
              </SelectItem>
              <SelectItem value="model">
                {t("settings.voice.dictation.engineModel")}
              </SelectItem>
            </SelectContent>
          </Select>
        </SettingsRow>

        {isLocalEngine ? (
          modelSttSupported ? (
            <SettingsRow
              label={t("settings.voice.dictation.sttModelLabel")}
              description={t("settings.voice.dictation.sttModelDescription")}
            >
              <div className="flex w-56 flex-col items-stretch gap-2">
                <SttModelPicker
                  value={sttModel}
                  language={dictationLanguage}
                  onChange={(next) => {
                    if (next !== sttModel) {
                      void unloadSttModel().catch(() => {});
                      void autoLoadSttModel(next);
                    }
                    setSttModel(next);
                  }}
                />
                {downloadingThisModel ? (
                  <div className="rounded-md border border-border/60 bg-muted/20 px-2.5 pt-2">
                    <div className="mb-1.5 flex items-center justify-between gap-3">
                      <span className="text-xs text-muted-foreground">
                        {sttModelStatusText}
                      </span>
                    </div>
                    <DownloadProgressBar
                      progress={{
                        expectedBytes: sttDownload?.bytes_total ?? 0,
                        downloadedBytes: sttDownload?.bytes_done ?? 0,
                        fraction:
                          sttDownload?.bytes_total && sttDownload.bytes_total > 0
                            ? (sttDownload.bytes_done ?? 0) /
                              sttDownload.bytes_total
                            : 0,
                      }}
                      bytesPerSec={downloadBytesPerSec}
                    />
                  </div>
                ) : (
                  <div className="flex min-h-7 items-center justify-between gap-3">
                    <span className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
                      {effectiveSttDownloadAvailability === "checking" ||
                      sttPhase === "loading" ||
                      sttPhase === "checking" ? (
                        <span className="size-1.5 shrink-0 animate-pulse rounded-full bg-current" />
                      ) : sttPhase === "ready" ||
                        (effectiveSttDownloadAvailability === "downloaded" &&
                          sttPhase === "on-demand") ? (
                        <span className="size-1.5 shrink-0 rounded-full bg-emerald-500" />
                      ) : effectiveSttDownloadAvailability === "error" ||
                        sttPhase === "error" ? (
                        <span className="size-1.5 shrink-0 rounded-full bg-destructive" />
                      ) : null}
                      <span>{sttModelStatusText}</span>
                    </span>
                    {sttPhase === "unavailable" ? (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        onClick={() => setStatusNonce((nonce) => nonce + 1)}
                      >
                        {t("settings.voice.dictation.sttRetry")}
                      </Button>
                    ) : effectiveSttDownloadAvailability === "error" ? (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        disabled={sttDownloadStarting || downloadingThisModel}
                        // Restart the download; the sidecar error is sticky
                        // until a new start(), so re-polling alone never clears it.
                        onClick={beginSttDownload}
                      >
                        {t("settings.voice.dictation.sttRetry")}
                      </Button>
                    ) : effectiveSttDownloadAvailability === "missing" ? (
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 px-2.5 text-xs"
                        disabled={sttDownloadStarting || downloadingThisModel}
                        onClick={beginSttDownload}
                      >
                        {sttDownloadStarting || downloadingThisModel ? (
                          <Spinner className="mr-1.5" />
                        ) : null}
                        {t("settings.voice.dictation.sttDownload")}
                      </Button>
                    ) : effectiveSttDownloadAvailability === "downloaded" ? (
                      sttPhase === "ready" ? (
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 px-2.5 text-xs"
                          disabled={sttUnloading}
                          onClick={releaseSttModel}
                        >
                          {sttUnloading ? <Spinner className="mr-1.5" /> : null}
                          {sttUnloading
                            ? t("settings.voice.dictation.sttUnloading")
                            : t("settings.voice.dictation.sttUnload")}
                        </Button>
                      ) : sttPhase === "loading" || sttPhase === "checking" ? (
                        // The status line already says "Loading model…";
                        // the button only needs the spinner.
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 px-2.5 text-xs"
                          disabled={true}
                          aria-label={t(
                            "settings.voice.dictation.sttLoadingModel",
                          )}
                        >
                          <Spinner />
                        </Button>
                      ) : (
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 px-2.5 text-xs"
                          onClick={warmSttModel}
                        >
                          {sttPhase === "error"
                            ? t("settings.voice.dictation.sttRetry")
                            : t("settings.voice.dictation.sttLoad")}
                        </Button>
                      )
                    ) : null}
                  </div>
                )}
              </div>
            </SettingsRow>
          ) : (
            <SettingsRow
              label={t("settings.voice.dictation.sttModelLabel")}
              description={t("settings.voice.dictation.sttModelUnsupported")}
            />
          )
        ) : null}

        <SettingsRow
          label={t("settings.voice.dictation.microphoneLabel")}
          description={
            hasLabels
              ? micDeviceId !== "default"
                ? t("settings.voice.dictation.microphoneFallbackHint")
                : t("settings.voice.dictation.microphoneDescription")
              : t("settings.voice.dictation.microphoneGrantDescription")
          }
        >
          {hasLabels ? (
            <Select value={micDeviceId} onValueChange={setMicDeviceId}>
              <SelectTrigger aria-label="Microphone" className="w-56" size="sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="default">
                  {t("settings.voice.dictation.systemDefault")}
                </SelectItem>
                {devices
                  .filter((d) => d.deviceId && d.deviceId !== "default")
                  .map((d, i) => (
                    <SelectItem key={d.deviceId} value={d.deviceId}>
                      {d.label || `Microphone ${i + 1}`}
                    </SelectItem>
                  ))}
                {!knownMic && micDeviceId !== "default" ? (
                  <SelectItem value={micDeviceId}>
                    {t("settings.voice.dictation.savedMicDisconnected")}
                  </SelectItem>
                ) : null}
              </SelectContent>
            </Select>
          ) : (
            <Button variant="outline" size="sm" onClick={requestAccess}>
              <MicIcon className="mr-1.5 size-3.5" />
              {t("settings.voice.dictation.allowMicrophone")}
            </Button>
          )}
        </SettingsRow>

        <SettingsRow
          label={t("settings.voice.dictation.languageLabel")}
          description={t("settings.voice.dictation.languageDescription")}
        >
          <Select
            value={dictationLanguage}
            onValueChange={setDictationLanguage}
          >
            <SelectTrigger
              aria-label="Dictation language"
              className="w-56"
              size="sm"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {DICTATION_LANGUAGES.map(({ value, label }) => (
                <SelectItem key={value} value={value}>
                  {value === "auto"
                    ? t("settings.voice.dictation.languageAuto")
                    : label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </SettingsRow>

        <SettingsRow
          label={t("settings.voice.dictionary.manageLabel")}
          description={t("settings.voice.dictionary.sectionDescription")}
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => setSubpage("dictionary")}
          >
            {t("settings.voice.dictionary.manage")}
          </Button>
        </SettingsRow>

        <SettingsRow
          label={t("settings.voice.recents.manageLabel")}
          description={t("settings.voice.recents.sectionDescription")}
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setSelectedDictationId(null);
              setSubpage("recents");
            }}
          >
            {t("settings.voice.recents.manage")}
          </Button>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.voice.readAloud.sectionTitle")}>
        {ttsSupported ? (
          <>
            <SettingsRow
              label={t("settings.voice.readAloud.buttonLabel")}
              description={t("settings.voice.readAloud.buttonDescription")}
            >
              <Switch checked={ttsEnabled} onCheckedChange={setTtsEnabled} />
            </SettingsRow>

            <SettingsRow
              label={t("settings.voice.readAloud.engineLabel")}
              description={
                effectiveTtsEngine === "studio"
                  ? t("settings.voice.readAloud.engineStudioDescription")
                  : t("settings.voice.readAloud.engineSystemDescription")
              }
            >
              <Select
                value={effectiveTtsEngine}
                onValueChange={(value) =>
                  setTtsEngine(value === "studio" ? "studio" : "system")
                }
              >
                <SelectTrigger
                  aria-label="TTS engine"
                  className="w-56"
                  size="sm"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {systemTtsSupported ? (
                    <SelectItem value="system">
                      {t("settings.voice.readAloud.engineSystem")}
                    </SelectItem>
                  ) : null}
                  <SelectItem value="studio">
                    {t("settings.voice.readAloud.engineStudio")}
                  </SelectItem>
                </SelectContent>
              </Select>
            </SettingsRow>

            {effectiveTtsEngine === "studio" ? (
              <SettingsRow
                label={t("settings.voice.readAloud.modelLabel")}
                description={t("settings.voice.readAloud.modelDescription")}
              />
            ) : (
              <SettingsRow
                label={t("settings.voice.readAloud.voiceLabel")}
                description={t("settings.voice.readAloud.voiceDescription")}
              >
                <Select value={ttsVoiceURI} onValueChange={setTtsVoiceURI}>
                  <SelectTrigger
                    aria-label="Text to speech voice"
                    className="w-56"
                    size="sm"
                  >
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="max-h-72">
                    <SelectItem value="default">
                      {t("settings.voice.dictation.systemDefault")}
                    </SelectItem>
                    {voices.map((voice) => (
                      <SelectItem key={voice.voiceURI} value={voice.voiceURI}>
                        {voice.name} ({voice.lang})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </SettingsRow>
            )}

            <SettingsRow
              label={t("settings.voice.readAloud.speedLabel")}
              description={`${ttsRate.toFixed(2)}x`}
            >
              <Slider
                value={[ttsRate]}
                min={0.5}
                max={2}
                step={0.05}
                onValueChange={([v]) => v !== undefined && setTtsRate(v)}
                className="w-48"
                aria-label="Speaking rate"
              />
            </SettingsRow>

            {effectiveTtsEngine === "system" && (
              <SettingsRow
                label={t("settings.voice.readAloud.pitchLabel")}
                description={`${ttsPitch.toFixed(2)}`}
              >
                <Slider
                  value={[ttsPitch]}
                  min={0}
                  max={2}
                  step={0.05}
                  onValueChange={([v]) => v !== undefined && setTtsPitch(v)}
                  className="w-48"
                  aria-label="Voice pitch"
                />
              </SettingsRow>
            )}

            <SettingsRow
              label={t("settings.voice.readAloud.volumeLabel")}
              description={`${Math.round(ttsVolume * 100)}%`}
            >
              <Slider
                value={[ttsVolume]}
                min={0}
                max={1}
                step={0.05}
                onValueChange={([v]) => v !== undefined && setTtsVolume(v)}
                className="w-48"
                aria-label="Playback volume"
              />
            </SettingsRow>

            <SettingsRow
              label={t("settings.voice.readAloud.previewLabel")}
              description={t("settings.voice.readAloud.previewDescription")}
            >
              <Button
                variant="outline"
                size="sm"
                onClick={() => void previewTts()}
              >
                {previewing ? (
                  <>
                    <SquareIcon className="mr-1.5 size-3 animate-pulse fill-current text-destructive" />
                    {t("settings.voice.readAloud.stopAction")}
                  </>
                ) : (
                  <>
                    <HugeiconsIcon
                      icon={VolumeHighIcon}
                      className="mr-1.5 size-3.5"
                    />
                    {t("settings.voice.readAloud.previewAction")}
                  </>
                )}
              </Button>
            </SettingsRow>
          </>
        ) : (
          <SettingsRow
            label={t("settings.voice.readAloud.ttsLabel")}
            description={t("settings.voice.readAloud.notSupported")}
          />
        )}
      </SettingsSection>
    </div>
  );
}
