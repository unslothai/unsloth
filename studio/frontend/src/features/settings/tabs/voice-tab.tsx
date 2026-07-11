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
import { StudioWebSpeechDictationAdapter } from "@/features/chat/adapters/studio-web-speech-dictation-adapter";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import {
  Copy01Icon,
  Delete02Icon,
  Mic02Icon,
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
  { value: "auto", label: "Auto (browser language)" },
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
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });
      stream.getTracks().forEach((track) => track.stop());
      await refresh();
    } catch {
      toast.error(
        "Microphone access was blocked. Allow microphone access for this Unsloth page, then try again.",
      );
    }
  }, [refresh]);

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
  const [testing, setTesting] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [interim, setInterim] = useState("");
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const stop = useCallback(() => {
    recognitionRef.current?.stop();
    recognitionRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setTesting(false);
    setInterim("");
  }, []);

  useEffect(() => stop, [stop]);

  const start = useCallback(async () => {
    const SpeechRecognitionAPI =
      window.SpeechRecognition ?? window.webkitSpeechRecognition;
    if (!SpeechRecognitionAPI) return;
    setTranscript("");
    setInterim("");

    const { micDeviceId } = useVoiceSettingsStore.getState();
    let audioTrack: MediaStreamTrack | undefined;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio:
          micDeviceId && micDeviceId !== "default"
            ? { deviceId: { exact: micDeviceId } }
            : true,
      });
      streamRef.current = stream;
      audioTrack = stream.getAudioTracks()[0];
    } catch {
      toast.error(
        "Could not open the selected microphone. Check permissions or pick another device.",
      );
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
          setTranscript((prev) => (prev ? `${prev} ${corrected}` : corrected));
        } else {
          interimText += text;
        }
      }
      setInterim(interimText);
    };
    recognition.onerror = () => stop();
    recognition.onend = () => {
      setTesting(false);
      setInterim("");
    };
    try {
      if (audioTrack) {
        try {
          recognition.start(audioTrack);
        } catch {
          recognition.start();
        }
      } else {
        recognition.start();
      }
    } catch {
      stop();
      return;
    }
    recognitionRef.current = recognition;
    setTesting(true);
  }, [stop]);

  const finishedTest = !testing && transcript;

  return (
    <div className="flex flex-col gap-2">
      <SettingsRow
        label="Test dictation"
        description="Speak to check your mic and settings"
      >
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            if (testing) {
              if (transcript) recordRecentDictation(transcript);
              stop();
            } else {
              void start();
            }
          }}
        >
          {testing ? (
            <>
              <SquareIcon className="mr-1.5 size-3 animate-pulse fill-current text-destructive" />
              Stop test
            </>
          ) : (
            <>
              <HugeiconsIcon icon={Mic02Icon} className="mr-1.5 size-3.5" />
              Start test
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
              {testing ? "Listening…" : ""}
            </span>
          )}
          {finishedTest ? (
            <div className="mt-1 text-xs text-muted-foreground">
              Saved to recent dictations
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

  const stopPreview = useCallback(() => {
    window.speechSynthesis?.cancel();
    previewAbortRef.current?.abort();
    previewAbortRef.current = null;
    if (previewAudioRef.current) {
      previewAudioRef.current.pause();
      previewAudioRef.current.src = "";
      previewAudioRef.current = null;
    }
    setPreviewing(false);
  }, []);

  const previewTts = async () => {
    if (!ttsSupported) return;
    if (previewing) {
      stopPreview();
      return;
    }
    if (ttsEngine === "studio") {
      const controller = new AbortController();
      previewAbortRef.current = controller;
      setPreviewing(true);
      try {
        const url = await generateStudioTtsAudio(
          TTS_PREVIEW_TEXT,
          controller.signal,
        );
        if (controller.signal.aborted) return;
        const audio = new Audio(url);
        audio.playbackRate = ttsRate;
        audio.volume = ttsVolume;
        audio.addEventListener("ended", () => setPreviewing(false));
        audio.addEventListener("error", () => setPreviewing(false));
        previewAudioRef.current = audio;
        await audio.play();
      } catch (error) {
        if (!controller.signal.aborted) {
          toast.error(
            error instanceof Error ? error.message : "TTS preview failed",
          );
        }
        setPreviewing(false);
      }
      return;
    }
    const utterance = createConfiguredUtterance(TTS_PREVIEW_TEXT);
    utterance.addEventListener("end", () => setPreviewing(false));
    utterance.addEventListener("error", () => setPreviewing(false));
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
    setPreviewing(true);
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

      <SettingsSection title="Dictation">
        <SettingsRow
          label="Microphone"
          description={
            hasLabels
              ? "Used for dictation"
              : "Allow mic access to show device names"
          }
        >
          {hasLabels || devices.length > 0 ? (
            <Select value={micDeviceId} onValueChange={setMicDeviceId}>
              <SelectTrigger aria-label="Microphone" className="w-56" size="sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="default">System default</SelectItem>
                {devices
                  .filter((d) => d.deviceId && d.deviceId !== "default")
                  .map((d, i) => (
                    <SelectItem key={d.deviceId} value={d.deviceId}>
                      {d.label || `Microphone ${i + 1}`}
                    </SelectItem>
                  ))}
                {!knownMic && micDeviceId !== "default" ? (
                  <SelectItem value={micDeviceId}>
                    Saved microphone (not connected)
                  </SelectItem>
                ) : null}
              </SelectContent>
            </Select>
          ) : (
            <Button variant="outline" size="sm" onClick={requestAccess}>
              <HugeiconsIcon icon={Mic02Icon} className="mr-1.5 size-3.5" />
              Allow microphone
            </Button>
          )}
        </SettingsRow>

        <SettingsRow
          label="Dictation language"
          description="Language to recognize"
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
                  {label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </SettingsRow>

        {dictationSupported ? (
          <DictationTest />
        ) : (
          <SettingsRow
            label="Test dictation"
            description="Not supported in this browser"
          />
        )}
      </SettingsSection>

      <SettingsSection
        title="Dictation dictionary"
        description="Words or phrases dictation should recognize"
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
              className="h-8 flex-1 text-sm"
              aria-label={`Dictionary entry ${index + 1}`}
            />
            <Button
              variant="ghost"
              size="icon"
              className="size-8 shrink-0 text-muted-foreground hover:text-destructive"
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
            Add entry
          </Button>
        </div>
      </SettingsSection>

      <SettingsSection
        title="Recent dictations"
        description="Your recent dictations will appear here so you can recover text"
      >
        {recentDictations.length === 0 ? (
          <p className="py-3 text-sm text-muted-foreground">
            No dictations yet
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
                    try {
                      await navigator.clipboard.writeText(item.text);
                      toast.success("Copied to clipboard");
                    } catch {
                      toast.error("Could not copy to clipboard");
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
                Clear recent dictations
              </Button>
            </div>
          </>
        )}
      </SettingsSection>

      <SettingsSection title="Read aloud">
        {ttsSupported ? (
          <>
            <SettingsRow
              label="Read aloud button"
              description="Show on assistant responses"
            >
              <Switch checked={ttsEnabled} onCheckedChange={setTtsEnabled} />
            </SettingsRow>

            <SettingsRow
              label="TTS engine"
              description={
                ttsEngine === "studio"
                  ? "Uses the loaded audio model (e.g. Orpheus)"
                  : "Built-in device voices"
              }
            >
              <Select
                value={ttsEngine}
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
                  <SelectItem value="system">System voices</SelectItem>
                  <SelectItem value="studio">Load TTS model</SelectItem>
                </SelectContent>
              </Select>
            </SettingsRow>

            {ttsEngine === "studio" ? (
              <SettingsRow
                label="TTS model"
                description="Load an audio model from the model selector (e.g. Orpheus TTS)"
              />
            ) : (
              <SettingsRow
                label="Voice"
                description="Best voices on this device"
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
                    <SelectItem value="default">System default</SelectItem>
                    {voices.map((voice) => (
                      <SelectItem key={voice.voiceURI} value={voice.voiceURI}>
                        {voice.name} ({voice.lang})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </SettingsRow>
            )}

            <SettingsRow label="Speed" description={`${ttsRate.toFixed(2)}x`}>
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

            {ttsEngine === "system" && (
              <SettingsRow label="Pitch" description={`${ttsPitch.toFixed(2)}`}>
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
              label="Volume"
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
              label="Preview voice"
              description="Play a short sample"
            >
              <Button
                variant="outline"
                size="sm"
                onClick={() => void previewTts()}
              >
                {previewing ? (
                  <>
                    <SquareIcon className="mr-1.5 size-3 animate-pulse fill-current text-destructive" />
                    Stop
                  </>
                ) : (
                  <>
                    <HugeiconsIcon
                      icon={VolumeHighIcon}
                      className="mr-1.5 size-3.5"
                    />
                    Preview
                  </>
                )}
              </Button>
            </SettingsRow>
          </>
        ) : (
          <SettingsRow
            label="Text to speech"
            description="Not supported in this browser"
          />
        )}
      </SettingsSection>
    </div>
  );
}
