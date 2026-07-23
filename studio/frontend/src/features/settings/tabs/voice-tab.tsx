// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  StudioSpeechSynthesisAdapter,
  createConfiguredUtterance,
  curateSystemVoices,
  generateStudioTtsAudio,
} from "@/features/chat/adapters/studio-speech-synthesis-adapter";
import {
  StudioWebSpeechDictationAdapter,
  describeSpeechError,
  isMissingDeviceError,
} from "@/features/chat/adapters/studio-web-speech-dictation-adapter";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { MicIcon } from "@/lib/mic-icon";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import {
  Copy01Icon,
  Delete02Icon,
  PlusSignIcon,
  VolumeHighIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { SquareIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import {
  applyDictationDictionary,
  recordRecentDictation,
  resolveDictationLanguage,
  useVoiceSettingsStore,
} from "../stores/voice-settings-store";

// Languages offered for browser speech recognition.
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
      stream.getTracks().forEach((track) => track.stop());
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

/** Inline mic test: runs speech recognition and shows the live transcript. */
function DictationTest() {
  const t = useT();
  const [testing, setTesting] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [interim, setInterim] = useState("");
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  // Guards the getUserMedia await so a mic opened after unmount is released.
  const disposedRef = useRef(false);
  // Mirrors the transcript state so onend can record it without stale closures.
  const transcriptRef = useRef("");

  // Single cleanup path: the browser can end recognition on its own (silence
  // timeout, service disconnect), so onend must release the mic and save the
  // transcript, not just the Stop button.
  const finalize = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    recognitionRef.current = null;
    if (transcriptRef.current) {
      recordRecentDictation(transcriptRef.current);
      transcriptRef.current = "";
    }
    setTesting(false);
    setInterim("");
  }, []);

  const stop = useCallback(() => {
    const recognition = recognitionRef.current;
    if (recognition) {
      // onend fires next and runs finalize()
      recognition.stop();
    } else {
      finalize();
    }
  }, [finalize]);

  useEffect(() => {
    disposedRef.current = false;
    return () => {
      disposedRef.current = true;
      recognitionRef.current?.abort();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, []);

  // Set before the getUserMedia await so a double click or a slow
  // permission prompt cannot start a second recognizer over the first.
  const startingRef = useRef(false);

  const start = useCallback(async () => {
    const SpeechRecognitionAPI =
      window.SpeechRecognition ?? window.webkitSpeechRecognition;
    if (!SpeechRecognitionAPI) return;
    if (startingRef.current || recognitionRef.current) return;
    startingRef.current = true;
    setTranscript("");
    setInterim("");
    transcriptRef.current = "";

    const { micDeviceId } = useVoiceSettingsStore.getState();
    let audioTrack: MediaStreamTrack | undefined;
    try {
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio:
            micDeviceId && micDeviceId !== "default"
              ? { deviceId: { exact: micDeviceId } }
              : true,
        });
      } catch (error) {
        // Saved mic may be unplugged; fall back to the default device.
        if (micDeviceId !== "default" && isMissingDeviceError(error)) {
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } else {
          throw error;
        }
      }
      if (disposedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        startingRef.current = false;
        return;
      }
      streamRef.current = stream;
      audioTrack = stream.getAudioTracks()[0];
    } catch {
      startingRef.current = false;
      toast.error(t("settings.voice.dictation.micOpenFailed"));
      return;
    }

    const recognition = new SpeechRecognitionAPI();
    recognition.lang = resolveDictationLanguage();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const text = result?.[0]?.transcript ?? "";
        if (result?.isFinal) {
          const corrected = applyDictationDictionary(text.trim());
          setTranscript((prev) => {
            const next = prev ? `${prev} ${corrected}` : corrected;
            transcriptRef.current = next;
            return next;
          });
        } else {
          interimText += text;
        }
      }
      setInterim(interimText);
    };
    recognition.onerror = (event) => {
      // onend follows and runs finalize(); surface non-abort failures here.
      const errorEvent = event as SpeechRecognitionErrorEvent;
      if (errorEvent.error !== "aborted") {
        toast.error(describeSpeechError(errorEvent.error, errorEvent.message));
      }
    };
    recognition.onend = () => finalize();
    try {
      if (audioTrack) {
        try {
          recognition.start(audioTrack);
        } catch {
          // Engine has no start(track) overload: it will capture from the
          // default device, so release the selected-device stream.
          streamRef.current?.getTracks().forEach((track) => track.stop());
          streamRef.current = null;
          recognition.start();
        }
      } else {
        recognition.start();
      }
    } catch {
      startingRef.current = false;
      finalize();
      return;
    }
    recognitionRef.current = recognition;
    startingRef.current = false;
    setTesting(true);
  }, [finalize, t]);

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
  const dictationLanguage = useVoiceSettingsStore((s) => s.dictationLanguage);
  const setDictationLanguage = useVoiceSettingsStore(
    (s) => s.setDictationLanguage,
  );
  const dictionary = useVoiceSettingsStore((s) => s.dictionary);
  const addDictionaryEntry = useVoiceSettingsStore((s) => s.addDictionaryEntry);
  const updateDictionaryEntry = useVoiceSettingsStore(
    (s) => s.updateDictionaryEntry,
  );
  const commitDictionaryEntry = useVoiceSettingsStore(
    (s) => s.commitDictionaryEntry,
  );
  const removeDictionaryEntry = useVoiceSettingsStore(
    (s) => s.removeDictionaryEntry,
  );
  const recentDictations = useVoiceSettingsStore((s) => s.recentDictations);
  const clearRecentDictations = useVoiceSettingsStore(
    (s) => s.clearRecentDictations,
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
    () => curateSystemVoices(rawVoices, ttsVoiceURI),
    // dictationLanguage feeds the curation language filter.
    [rawVoices, ttsVoiceURI, dictationLanguage],
  );
  const [newEntry, setNewEntry] = useState("");
  const [previewing, setPreviewing] = useState(false);

  const dictationSupported = StudioWebSpeechDictationAdapter.isSupported();
  const ttsSupported = StudioSpeechSynthesisAdapter.isSupported();
  const systemTtsSupported =
    StudioSpeechSynthesisAdapter.systemVoicesSupported();
  const effectiveTtsEngine = systemTtsSupported ? ttsEngine : "studio";

  // Keep an item for an unplugged saved mic so the value stays visible.
  const knownMic = devices.some((d) => d.deviceId === micDeviceId);

  const handleAddEntry = () => {
    const trimmed = newEntry.trim();
    if (!trimmed) return;
    addDictionaryEntry(trimmed);
    setNewEntry("");
  };

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
        // Some browsers reset playbackRate to 1 once the source loads; reapply
        // it on loadedmetadata so the speed setting reliably takes effect.
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
          // Surface playback failures like the catch below, instead of just
          // resetting the button with no explanation.
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
              <SelectTrigger
                aria-label="Microphone"
                className="min-w-56 max-w-72"
                size="sm"
              >
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
              className="min-w-56 max-w-72"
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
      </SettingsSection>

      <SettingsSection
        title={t("settings.voice.dictionary.sectionTitle")}
        description={t("settings.voice.dictionary.sectionDescription")}
      >
        {dictionary.map((entry, index) => (
          <div
            // biome-ignore lint/suspicious/noArrayIndexKey: entries are editable in place
            key={index}
            className="flex items-center gap-2 py-1.5"
          >
            <Input
              value={entry}
              onChange={(e) => updateDictionaryEntry(index, e.target.value)}
              // Skip the empty-row commit-splice when focus moves to this row's
              // Remove button (keyboard Tab), so its index stays valid and its
              // activation deletes this row instead of the next one.
              onBlur={(e) => {
                if (
                  (e.relatedTarget as HTMLElement | null)?.dataset.dictRemove ===
                  String(index)
                ) {
                  return;
                }
                commitDictionaryEntry(index);
              }}
              className="h-8 flex-1 text-sm"
              aria-label={`Dictionary entry ${index + 1}`}
            />
            <Button
              variant="ghost"
              size="icon"
              className="size-8 shrink-0 text-muted-foreground hover:text-destructive"
              data-dict-remove={index}
              // Mouse: keep the click from blurring an empty input first, which
              // would commit-splice this row and make onClick delete the next.
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => removeDictionaryEntry(index)}
              aria-label={`Remove dictionary entry ${index + 1}`}
            >
              <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
            </Button>
          </div>
        ))}
        <div className="flex items-center gap-2 py-1.5">
          <Input
            value={newEntry}
            onChange={(e) => setNewEntry(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleAddEntry();
              }
            }}
            placeholder="Jane Doe"
            className="h-8 flex-1 text-sm"
            aria-label="New dictionary entry"
          />
          <Button
            variant="outline"
            size="sm"
            className="shrink-0"
            onClick={handleAddEntry}
            disabled={!newEntry.trim()}
          >
            <HugeiconsIcon icon={PlusSignIcon} className="mr-1.5 size-3.5" />
            {t("settings.voice.dictionary.addEntry")}
          </Button>
        </div>
      </SettingsSection>

      <SettingsSection
        title={t("settings.voice.recents.sectionTitle")}
        description={t("settings.voice.recents.sectionDescription")}
      >
        {recentDictations.length === 0 ? (
          <p className="py-3 text-sm text-muted-foreground">
            {t("settings.voice.recents.empty")}
          </p>
        ) : (
          <>
            {recentDictations.map((item) => (
              <div
                key={`${item.at}-${item.text.slice(0, 24)}`}
                className="flex items-start justify-between gap-3 py-2.5"
              >
                <div className="min-w-0 flex-1">
                  <p className="whitespace-pre-wrap break-words text-sm text-foreground">
                    {item.text}
                  </p>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    {new Date(item.at).toLocaleString()}
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-8 shrink-0 text-muted-foreground"
                  aria-label="Copy dictation"
                  onClick={async () => {
                    // Helper falls back to execCommand where navigator.clipboard
                    // is unavailable (Safari, insecure http LAN contexts).
                    if (await copyToClipboard(item.text)) {
                      toast.success(t("settings.voice.recents.copied"));
                    } else {
                      toast.error(t("settings.voice.recents.copyFailed"));
                    }
                  }}
                >
                  <HugeiconsIcon icon={Copy01Icon} className="size-3.5" />
                </Button>
              </div>
            ))}
            <div className="flex justify-end py-2">
              <Button
                variant="outline"
                size="sm"
                onClick={clearRecentDictations}
                className="text-destructive hover:text-destructive hover:border-destructive/60"
              >
                <HugeiconsIcon
                  icon={Delete02Icon}
                  className="mr-1.5 size-3.5"
                />
                {t("settings.voice.recents.clear")}
              </Button>
            </div>
          </>
        )}
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
                  className="min-w-56 max-w-72"
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
                    className="min-w-56 max-w-72"
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
