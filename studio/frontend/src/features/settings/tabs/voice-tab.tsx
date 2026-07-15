// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
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
import { StudioDictationAdapter } from "@/features/chat/adapters/studio-dictation-adapter";
import {
  StudioModelDictationAdapter,
  fetchSttStatus,
  loadSttModel,
  unloadSttModel,
} from "@/features/chat/adapters/studio-model-dictation-adapter";
import {
  StudioSpeechSynthesisAdapter,
  createConfiguredUtterance,
  curateSystemVoices,
  generateStudioTtsAudio,
} from "@/features/chat/adapters/studio-speech-synthesis-adapter";
import type { StudioDictationSession } from "@/features/chat/adapters/studio-web-speech-dictation-adapter";
import {
  DownloadProgressBar,
  TransportConflictDialog,
  getDownloadProgress,
  useRepoDownload,
} from "@/features/hub";
import { useHubModelSearch } from "@/features/hub/hooks/use-hub-model-search";
import {
  hfApiToken,
  useHfTokenStore,
} from "@/features/hub/stores/hf-token-store";
import { useDebouncedValue } from "@/hooks";
import { useT } from "@/i18n";
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
const STT_MODEL_SIZES: Record<DefaultSttModel, string> = {
  tiny: "155 MB",
  base: "295 MB",
  small: "970 MB",
  "large-v3-turbo": "1.6 GB",
  "large-v3": "3.1 GB",
};

function sttModelName(model: SttModel): string {
  return STT_MODEL_NAMES[model as DefaultSttModel] ?? model;
}

function sttModelSize(model: SttModel): string {
  return STT_MODEL_SIZES[model as DefaultSttModel] ?? "";
}

function SttModelCombobox({
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
  const anchorRef = useRef<HTMLDivElement>(null);
  // Set while a pick is in flight so the input keeps the model's display text
  // instead of the combobox writing the raw value back as a search query.
  const selectingRef = useRef(false);
  const [inputValue, setInputValue] = useState(() => sttModelName(value));
  // A pick fills the input with the model's display text. Treat that text as a
  // selection, not a search, so choosing a model never triggers a lookup.
  const trimmedInput = inputValue.trim();
  const isSelectedText =
    trimmedInput === sttModelName(value) ||
    STT_MODELS.some((model) => sttModelName(model) === trimmedInput);
  const query = isSelectedText ? "" : trimmedInput;
  const debouncedQuery = useDebouncedValue(query);

  useEffect(() => {
    setInputValue(sttModelName(value));
  }, [value]);

  const { results, isLoading } = useHubModelSearch(debouncedQuery, {
    task: "automatic-speech-recognition",
    accessToken: hfApiToken(hfToken),
    excludeGguf: true,
    enabled: debouncedQuery.length >= 2,
    keepUnsupportedTags: true,
    ownerScope: "all",
  });

  const items = useMemo(() => {
    if (!query) {
      const defaults: string[] = STT_MODELS.filter((model) =>
        isSttModelLanguageCompatible(model, language),
      );
      if (!defaults.includes(value)) {
        defaults.push(value);
      }
      return defaults;
    }

    const ids: string[] = [];
    if (isSttModelId(query)) {
      ids.push(query);
    }
    for (const result of results) {
      const tags = result.tags?.map((tag) => tag.toLowerCase()) ?? [];
      const isWhisper =
        result.id.toLowerCase().includes("whisper") || tags.includes("whisper");
      if (
        isWhisper &&
        result.pipelineTag === "automatic-speech-recognition" &&
        isSttModelLanguageCompatible(result.id, language) &&
        !ids.includes(result.id)
      ) {
        ids.push(result.id);
      }
    }
    if (!ids.includes(value)) {
      ids.push(value);
    }
    return ids;
  }, [language, query, results, value]);

  const selectModel = (model: string | null) => {
    if (!(model && isSttModelId(model))) {
      return;
    }
    selectingRef.current = true;
    onChange(model);
    setInputValue(sttModelName(model));
  };

  return (
    <div
      ref={anchorRef}
      onKeyDown={(event) => {
        if (event.key !== "Enter") {
          return;
        }
        if (!(event.target instanceof HTMLInputElement)) {
          return;
        }
        if (!query || items.length === 0) {
          return;
        }
        event.preventDefault();
        selectModel(items[0]);
      }}
    >
      <Combobox
        items={items}
        filteredItems={items}
        filter={null}
        value={value}
        inputValue={inputValue}
        onValueChange={selectModel}
        onInputValueChange={(next) => {
          // Ignore the value the combobox echoes back on a pick; keep our display.
          if (selectingRef.current) {
            selectingRef.current = false;
            return;
          }
          setInputValue(next);
        }}
        itemToStringLabel={sttModelName}
        itemToStringValue={sttModelName}
        autoHighlight={true}
      >
        <ComboboxInput
          aria-label="Speech recognition model"
          placeholder={t("settings.voice.dictation.sttModelSearchPlaceholder")}
          className="h-8 w-full [&_input]:text-sm"
          startAddon={
            <HugeiconsIcon
              icon={Search01Icon}
              strokeWidth={2}
              className="size-3.5 text-muted-foreground"
            />
          }
        />
        <ComboboxContent anchor={anchorRef}>
          {isLoading && query ? (
            <div className="flex items-center gap-2 px-3 py-3 text-xs text-muted-foreground">
              <Spinner className="size-3.5" />
              {t("settings.voice.dictation.sttModelSearching")}
            </div>
          ) : (
            <ComboboxEmpty>
              {t("settings.voice.dictation.sttModelNoResults")}
            </ComboboxEmpty>
          )}
          <ComboboxList>
            {(model: string) => {
              const curated = (STT_MODELS as readonly string[]).includes(model);
              return (
                <ComboboxItem
                  key={model}
                  value={model}
                  onPointerDown={() => {
                    selectingRef.current = true;
                  }}
                >
                  <span className="min-w-0 flex-1 truncate">
                    <span className="block truncate text-xs">
                      {sttModelName(model)}
                    </span>
                    {curated ? (
                      <span className="mt-0.5 block truncate font-mono text-[9px] leading-tight text-muted-foreground">
                        {getSttModelRepo(model)}
                      </span>
                    ) : null}
                  </span>
                  {sttModelSize(model) ? (
                    <span className="shrink-0 text-[10px] tabular-nums text-muted-foreground">
                      {sttModelSize(model)}
                    </span>
                  ) : null}
                </ComboboxItem>
              );
            }}
          </ComboboxList>
        </ComboboxContent>
      </Combobox>
    </div>
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

/** Inline mic test: runs dictation on the selected engine and shows the text. */
function DictationTest() {
  const t = useT();
  const [testing, setTesting] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [interim, setInterim] = useState("");
  const sessionRef = useRef<StudioDictationSession | null>(null);
  // Set before the listen() call so a double click cannot start two sessions.
  const startingRef = useRef(false);

  const stop = useCallback(() => {
    // onEnd resets state once the session (and any transcription) finishes.
    void sessionRef.current?.stop();
  }, []);

  useEffect(
    () => () => {
      sessionRef.current?.cancel();
      sessionRef.current = null;
    },
    [],
  );

  const start = useCallback(async () => {
    if (startingRef.current || sessionRef.current) return;
    if (!StudioDictationAdapter.isSupported()) return;
    startingRef.current = true;
    setTranscript("");
    setInterim("");

    let session: StudioDictationSession;
    try {
      // Routes to the browser or STT-model engine; the adapter applies the
      // dictionary and saves the result to Recent dictations.
      session = new StudioDictationAdapter().listen();
    } catch {
      startingRef.current = false;
      toast.error(t("settings.voice.dictation.micOpenFailed"));
      return;
    }
    sessionRef.current = session;
    setTesting(true);
    session.onSpeech((result) => {
      const text = result.transcript?.trim() ?? "";
      if (result.isFinal) {
        setInterim("");
        if (text) setTranscript((prev) => (prev ? `${prev} ${text}` : text));
      } else {
        setInterim(result.transcript ?? "");
      }
    });
    session.onEnd?.(() => {
      if (sessionRef.current === session) sessionRef.current = null;
      setTesting(false);
      setInterim("");
    });
    startingRef.current = false;
  }, [t]);

  const finishedTest = !testing && transcript;

  return (
    <div className="flex flex-col gap-2">
      <SettingsRow
        label={t("settings.voice.dictation.testLabel")}
        description={t("settings.voice.dictation.testDescription")}
      >
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            if (testing) {
              stop();
            } else {
              void start();
            }
          }}
        >
          {testing ? (
            <>
              <SquareIcon className="mr-1.5 size-3 animate-pulse fill-current text-destructive" />
              {t("settings.voice.dictation.stopTest")}
            </>
          ) : (
            <>
              <MicIcon className="mr-1.5 size-3.5" />
              {t("settings.voice.dictation.startTest")}
            </>
          )}
        </Button>
      </SettingsRow>
      {(testing || transcript) && (
        <div className="rounded-lg border border-border/60 bg-muted/30 px-3 py-2 text-sm">
          {transcript || interim ? (
            <>
              <span className="text-foreground">{transcript}</span>
              {interim ? (
                <span className="text-muted-foreground"> {interim}</span>
              ) : null}
            </>
          ) : (
            <span className="text-muted-foreground">
              {testing ? t("settings.voice.dictation.listening") : ""}
            </span>
          )}
          {finishedTest ? (
            <div className="mt-1 text-xs text-muted-foreground">
              {t("settings.voice.dictation.testSaved")}
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
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

  const dictationSupported = StudioDictationAdapter.isSupported();
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
  const sttDownload = useRepoDownload({
    kind: "model",
    repoId: sttRepoId,
    activeVariant: null,
    autoAdopt: dictationEngine === "model" && modelSttSupported,
    onComplete: () => {
      setSttDownloadAvailability({
        repoId: sttRepoId,
        state: "downloaded",
      });
      toast.success(t("settings.voice.dictation.sttDownloadComplete"));
    },
    onCancelled: () => {
      setSttDownloadAvailability({ repoId: sttRepoId, state: "missing" });
    },
    onError: () => {
      setSttDownloadAvailability({ repoId: sttRepoId, state: "error" });
      toast.error(t("settings.voice.dictation.sttDownloadFailed"));
    },
  });

  const checkSttDownloadAvailability = useCallback(
    async (signal?: AbortSignal) => {
      try {
        const progress = await getDownloadProgress(sttRepoId, {
          hfToken: hfApiToken(hfToken),
          signal,
        });
        if (signal?.aborted) {
          return;
        }
        setSttDownloadAvailability({
          repoId: sttRepoId,
          state: progress.complete_on_disk ? "downloaded" : "missing",
        });
      } catch {
        if (!signal?.aborted) {
          setSttDownloadAvailability({ repoId: sttRepoId, state: "error" });
        }
      }
    },
    [sttRepoId, hfToken],
  );

  useEffect(() => {
    if (dictationEngine !== "model" || !modelSttSupported) {
      return;
    }
    const controller = new AbortController();
    checkSttDownloadAvailability(controller.signal).catch(() => {
      // The checker records a user-visible error state itself.
    });
    return () => controller.abort();
  }, [checkSttDownloadAvailability, dictationEngine, modelSttSupported]);

  useEffect(() => {
    if (dictationEngine !== "model" || !modelSttSupported) {
      setSttPhase("idle");
      setSttDevice(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      setSttPhase("checking");
      try {
        const status = await fetchSttStatus(statusNonce);
        if (cancelled) return;
        if (!status.available) {
          setSttPhase("unavailable");
          return;
        }
        if (status.loading) {
          setSttPhase("loading");
          window.setTimeout(() => {
            if (!cancelled) setStatusNonce((n) => n + 1);
          }, 600);
          return;
        }
        if (status.loaded_model === sttModel && !status.loading) {
          setSttDevice(status.device);
          setSttPhase("ready");
          window.setTimeout(
            () => {
              if (!cancelled) setStatusNonce((n) => n + 1);
            },
            Math.max(1000, Math.min(status.keep_alive_seconds * 1000, 15_000)),
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
  }, [dictationEngine, sttModel, modelSttSupported, statusNonce]);

  const sttStatusText = (() => {
    switch (sttPhase) {
      case "checking":
        return t("settings.voice.dictation.sttChecking");
      case "loading":
        return t("settings.voice.dictation.sttLoadingModel");
      case "on-demand":
        return t("settings.voice.dictation.sttOnDemand");
      case "ready":
        return t("settings.voice.dictation.sttReady", {
          device: (sttDevice ?? "cpu").toUpperCase(),
        });
      case "unavailable":
        return t("settings.voice.dictation.sttUnavailable");
      case "error":
        return t("settings.voice.dictation.sttModelFailed");
      default:
        return "";
    }
  })();

  const sttModelStatusText = (() => {
    if (sttDownload.progress) {
      return t("settings.voice.dictation.sttDownloading", {
        progress: Math.round(sttDownload.progress.fraction * 100),
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

  const startSttDownload = async () => {
    setSttDownloadStarting(true);
    try {
      await sttDownload.requestStartDownload(null, 0);
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
        audio.addEventListener("ended", () => {
          releasePreviewAudio();
          markPreviewing(false);
        });
        audio.addEventListener("error", () => {
          releasePreviewAudio();
          markPreviewing(false);
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
              if (next === "browser") {
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

        {dictationEngine === "model" ? (
          modelSttSupported ? (
            <SettingsRow
              label={t("settings.voice.dictation.sttModelLabel")}
              description={t("settings.voice.dictation.sttModelDescription")}
            >
              <div className="flex w-56 flex-col items-stretch gap-2">
                <SttModelCombobox
                  value={sttModel}
                  language={dictationLanguage}
                  onChange={(next) => {
                    if (next !== sttModel) {
                      void unloadSttModel().catch(() => {});
                    }
                    setSttModel(next);
                  }}
                />
                {sttDownload.progress ? (
                  <div className="rounded-md border border-border/60 bg-muted/20 px-2.5 pt-2">
                    <div className="mb-1.5 flex items-center justify-between gap-3">
                      <span className="text-xs text-muted-foreground">
                        {sttModelStatusText}
                      </span>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-6 px-2 text-xs"
                        disabled={sttDownload.cancelling}
                        onClick={() => sttDownload.cancelDownload(null)}
                      >
                        {sttDownload.cancelling
                          ? t("settings.voice.dictation.sttCancellingDownload")
                          : t("settings.voice.dictation.sttCancelDownload")}
                      </Button>
                    </div>
                    <DownloadProgressBar
                      progress={sttDownload.progress}
                      bytesPerSec={sttDownload.bytesPerSec}
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
                        onClick={async () => {
                          setSttDownloadAvailability({
                            repoId: sttRepoId,
                            state: "checking",
                          });
                          await checkSttDownloadAvailability();
                        }}
                      >
                        {t("settings.voice.dictation.sttRetry")}
                      </Button>
                    ) : effectiveSttDownloadAvailability === "missing" ? (
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 px-2.5 text-xs"
                        disabled={sttDownloadStarting}
                        onClick={startSttDownload}
                      >
                        {sttDownloadStarting ? (
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
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-7 px-2.5 text-xs"
                          disabled={true}
                        >
                          <Spinner className="mr-1.5" />
                          {t("settings.voice.dictation.sttLoadingModel")}
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
                <TransportConflictDialog
                  conflict={sttDownload.transportConflict}
                  onCancel={sttDownload.cancelConflict}
                  onKeepTransport={sttDownload.resumeConflict}
                  onSwitchTransport={sttDownload.restartConflict}
                />
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

        {dictationSupported ? (
          <DictationTest />
        ) : (
          <SettingsRow
            label={t("settings.voice.dictation.testLabel")}
            description={t("settings.voice.dictation.notSupported")}
          />
        )}
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
