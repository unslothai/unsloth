// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Settings section for system-wide voice typing. Rendered only in the desktop app
// on platforms that can arm a global shortcut (see voice_typing_info).

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { useMicrophonePermissionStore } from "@/features/microphone";
import {
  MIC_WARM_ALWAYS,
  VOICE_TYPING_SOUND_THEMES,
  type VoiceTypingMode,
  type VoiceTypingSoundTheme,
  chordFromEvent,
  formatShortcut,
  isModifierCode,
  previewVoiceTypingSound,
  useVoiceTypingStore,
} from "@/features/voice-typing";
import { type TranslationKey, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { invoke } from "@tauri-apps/api/core";
import { Play } from "lucide-react";
import { useEffect, useState } from "react";
import { SettingsRow } from "./settings-row";
import { SettingsGroupDivider, SettingsSection } from "./settings-section";

interface VoiceTypingInfo {
  os: string;
  shortcutSupported: boolean;
  holdSupported: boolean;
  pasteSupported: boolean;
}

const SOUND_THEME_LABEL: Record<VoiceTypingSoundTheme, TranslationKey> = {
  chime: "settings.voice.voiceTyping.soundsThemeChime",
  marimba: "settings.voice.voiceTyping.soundsThemeMarimba",
  pop: "settings.voice.voiceTyping.soundsThemePop",
  soft: "settings.voice.voiceTyping.soundsThemeSoft",
};

export function VoiceTypingSettingsSection() {
  const t = useT();
  const enabled = useVoiceTypingStore((state) => state.enabled);
  const setEnabled = useVoiceTypingStore((state) => state.setEnabled);
  const shortcut = useVoiceTypingStore((state) => state.shortcut);
  const setShortcut = useVoiceTypingStore((state) => state.setShortcut);
  const mode = useVoiceTypingStore((state) => state.mode);
  const setMode = useVoiceTypingStore((state) => state.setMode);
  const soundEnabled = useVoiceTypingStore((state) => state.soundEnabled);
  const setSoundEnabled = useVoiceTypingStore((state) => state.setSoundEnabled);
  const soundTheme = useVoiceTypingStore((state) => state.soundTheme);
  const setSoundTheme = useVoiceTypingStore((state) => state.setSoundTheme);
  const soundVolume = useVoiceTypingStore((state) => state.soundVolume);
  const setSoundVolume = useVoiceTypingStore((state) => state.setSoundVolume);
  const appendTrailingSpace = useVoiceTypingStore(
    (state) => state.appendTrailingSpace,
  );
  const setAppendTrailingSpace = useVoiceTypingStore(
    (state) => state.setAppendTrailingSpace,
  );
  const copyLastTranscript = useVoiceTypingStore(
    (state) => state.copyLastTranscript,
  );
  const setCopyLastTranscript = useVoiceTypingStore(
    (state) => state.setCopyLastTranscript,
  );
  const preloadModel = useVoiceTypingStore((state) => state.preloadModel);
  const setPreloadModel = useVoiceTypingStore((state) => state.setPreloadModel);
  const autoStopSilence = useVoiceTypingStore((state) => state.autoStopSilence);
  const setAutoStopSilence = useVoiceTypingStore(
    (state) => state.setAutoStopSilence,
  );
  const autoStopDelayMs = useVoiceTypingStore((state) => state.autoStopDelayMs);
  const setAutoStopDelayMs = useVoiceTypingStore(
    (state) => state.setAutoStopDelayMs,
  );
  const micWarmSeconds = useVoiceTypingStore((state) => state.micWarmSeconds);
  const setMicWarmSeconds = useVoiceTypingStore(
    (state) => state.setMicWarmSeconds,
  );
  const removeFillerWords = useVoiceTypingStore(
    (state) => state.removeFillerWords,
  );
  const setRemoveFillerWords = useVoiceTypingStore(
    (state) => state.setRemoveFillerWords,
  );
  const spokenPunctuation = useVoiceTypingStore(
    (state) => state.spokenPunctuation,
  );
  const setSpokenPunctuation = useVoiceTypingStore(
    (state) => state.setSpokenPunctuation,
  );
  const submitAfterInsert = useVoiceTypingStore(
    (state) => state.submitAfterInsert,
  );
  const setSubmitAfterInsert = useVoiceTypingStore(
    (state) => state.setSubmitAfterInsert,
  );
  const pasteAutomatically = useVoiceTypingStore(
    (state) => state.pasteAutomatically,
  );
  const setPasteAutomatically = useVoiceTypingStore(
    (state) => state.setPasteAutomatically,
  );

  const micPermission = useMicrophonePermissionStore(
    (state) => state.permission,
  );
  const loadMicPermission = useMicrophonePermissionStore((state) => state.load);
  const requestMicPermission = useMicrophonePermissionStore(
    (state) => state.request,
  );
  const askAgainMicPermission = useMicrophonePermissionStore(
    (state) => state.askAgain,
  );
  const resetMicPermission = useMicrophonePermissionStore(
    (state) => state.reset,
  );

  const [info, setInfo] = useState<VoiceTypingInfo | null>(null);
  const [capturing, setCapturing] = useState(false);
  const [needsModifier, setNeedsModifier] = useState(false);

  useEffect(() => {
    if (!isTauri) return;
    let live = true;
    void invoke<VoiceTypingInfo>("voice_typing_info")
      .then((value) => {
        if (live) setInfo(value);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    void loadMicPermission();
  }, [loadMicPermission]);

  // While the capture box is armed, the next chord becomes the shortcut. Capture
  // runs before the app's own shortcut handling so nothing else reacts.
  useEffect(() => {
    if (!capturing) return;
    const onKeyDown = (event: KeyboardEvent) => {
      event.preventDefault();
      event.stopPropagation();
      if (event.key === "Escape") {
        setCapturing(false);
        setNeedsModifier(false);
        return;
      }
      const chord = chordFromEvent(event);
      if (!chord) {
        // Modifier-only presses keep waiting; a bare key explains the rule.
        if (!isModifierCode(event.code)) setNeedsModifier(true);
        return;
      }
      setShortcut(chord);
      setCapturing(false);
      setNeedsModifier(false);
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [capturing, setShortcut]);

  // Browser dev sessions and platforms without a shortcut layer never see this.
  if (!isTauri || !info?.shortcutSupported) return null;

  const modeDescription =
    mode === "hold"
      ? t("settings.voice.voiceTyping.modeHoldDescription")
      : mode === "toggle"
        ? t("settings.voice.voiceTyping.modeToggleDescription")
        : t("settings.voice.voiceTyping.modeHybridDescription");

  const micStatusText =
    micPermission === "granted"
      ? t("settings.voice.voiceTyping.micAccessGranted")
      : micPermission === "denied"
        ? t("settings.voice.voiceTyping.micAccessBlockedStatus")
        : t("settings.voice.voiceTyping.micAccessNeeded");
  const micStatusDescription =
    micPermission === "granted"
      ? t("settings.voice.voiceTyping.micAccessGrantedDescription")
      : micPermission === "denied"
        ? t("settings.voice.voiceTyping.micAccessBlockedDescription")
        : t("settings.voice.voiceTyping.micAccessNeededDescription");

  return (
    <SettingsSection title={t("settings.voice.voiceTyping.sectionTitle")}>
      <SettingsRow
        label={t("settings.voice.voiceTyping.enableLabel")}
        description={t("settings.voice.voiceTyping.enableDescription")}
      >
        <Switch
          checked={enabled}
          onCheckedChange={setEnabled}
          data-testid="voice-typing-enable"
          aria-label={t("settings.voice.voiceTyping.enableLabel")}
        />
      </SettingsRow>

      {enabled ? (
        <>
          <p
            className="-mt-1 pb-3 text-xs text-muted-foreground"
            data-testid="voice-typing-active-hint"
          >
            {t("settings.voice.voiceTyping.activeHint", {
              shortcut: formatShortcut(shortcut),
            })}
          </p>

          <SettingsGroupDivider />

          <SettingsRow
            label={t("settings.voice.voiceTyping.shortcutLabel")}
            description={t("settings.voice.voiceTyping.shortcutHint")}
          >
            <Button
              variant="outline"
              size="sm"
              data-testid="voice-typing-shortcut"
              className="min-w-44 justify-center font-mono text-xs"
              aria-label={t("settings.voice.voiceTyping.shortcutLabel")}
              onClick={() => {
                setNeedsModifier(false);
                setCapturing((value) => !value);
              }}
            >
              {capturing
                ? t("settings.voice.voiceTyping.shortcutListening")
                : formatShortcut(shortcut)}
            </Button>
          </SettingsRow>
          {needsModifier ? (
            <p className="pb-2 text-xs text-destructive">
              {t("settings.voice.voiceTyping.shortcutNeedsModifier")}
            </p>
          ) : null}

          <SettingsRow
            label={t("settings.voice.voiceTyping.modeLabel")}
            description={modeDescription}
          >
            <Select
              value={mode}
              onValueChange={(value) => setMode(value as VoiceTypingMode)}
            >
              <SelectTrigger
                size="sm"
                className="w-44"
                data-testid="voice-typing-mode"
                aria-label={t("settings.voice.voiceTyping.modeLabel")}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="hybrid">
                  {t("settings.voice.voiceTyping.modeHybrid")}
                </SelectItem>
                <SelectItem value="hold">
                  {t("settings.voice.voiceTyping.modeHold")}
                </SelectItem>
                <SelectItem value="toggle">
                  {t("settings.voice.voiceTyping.modeToggle")}
                </SelectItem>
              </SelectContent>
            </Select>
          </SettingsRow>

          <SettingsGroupDivider />

          <SettingsRow
            label={t("settings.voice.voiceTyping.micAccessLabel")}
            description={micStatusDescription}
          >
            <div className="flex items-center gap-2">
              <span
                className="flex items-center gap-1.5 text-xs text-muted-foreground"
                data-testid="voice-typing-mic-status"
              >
                <span
                  className={`size-1.5 rounded-full ${
                    micPermission === "granted"
                      ? "bg-emerald-500"
                      : micPermission === "denied"
                        ? "bg-destructive"
                        : "bg-muted-foreground/50"
                  }`}
                />
                {micStatusText}
              </span>
              {micPermission === "granted" ? (
                <Button
                  variant="ghost"
                  size="sm"
                  data-testid="voice-typing-mic-reset"
                  onClick={() => {
                    void resetMicPermission();
                  }}
                >
                  {t("settings.voice.voiceTyping.micAccessReset")}
                </Button>
              ) : micPermission === "denied" ? (
                <Button
                  variant="outline"
                  size="sm"
                  data-testid="voice-typing-mic-ask-again"
                  onClick={() => {
                    void askAgainMicPermission("settings");
                  }}
                >
                  {t("settings.voice.voiceTyping.micAccessAskAgain")}
                </Button>
              ) : (
                <Button
                  variant="outline"
                  size="sm"
                  data-testid="voice-typing-mic-allow"
                  onClick={() => {
                    void requestMicPermission("settings");
                  }}
                >
                  {t("settings.voice.dictation.allowMicrophone")}
                </Button>
              )}
            </div>
          </SettingsRow>

          <SettingsRow
            label={t("settings.voice.voiceTyping.micWarmLabel")}
            description={
              micWarmSeconds === MIC_WARM_ALWAYS
                ? t("settings.voice.voiceTyping.micWarmAlwaysDescription")
                : t("settings.voice.voiceTyping.micWarmDescription")
            }
          >
            <Select
              value={String(micWarmSeconds)}
              onValueChange={(value) => setMicWarmSeconds(Number(value))}
            >
              <SelectTrigger
                size="sm"
                className="w-44"
                data-testid="voice-typing-mic-warm"
                aria-label={t("settings.voice.voiceTyping.micWarmLabel")}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0">
                  {t("settings.voice.voiceTyping.micWarmOff")}
                </SelectItem>
                <SelectItem value="15">
                  {t("settings.voice.voiceTyping.micWarmSeconds", {
                    seconds: 15,
                  })}
                </SelectItem>
                <SelectItem value="30">
                  {t("settings.voice.voiceTyping.micWarmSeconds", {
                    seconds: 30,
                  })}
                </SelectItem>
                <SelectItem value="60">
                  {t("settings.voice.voiceTyping.micWarmSeconds", {
                    seconds: 60,
                  })}
                </SelectItem>
                <SelectItem value={String(MIC_WARM_ALWAYS)}>
                  {t("settings.voice.voiceTyping.micWarmAlways")}
                </SelectItem>
              </SelectContent>
            </Select>
          </SettingsRow>

          <SettingsGroupDivider />

          <SettingsRow
            label={t("settings.voice.voiceTyping.pasteLabel")}
            description={t("settings.voice.voiceTyping.pasteDescription")}
          >
            <Switch
              checked={pasteAutomatically}
              onCheckedChange={setPasteAutomatically}
              data-testid="voice-typing-paste"
              aria-label={t("settings.voice.voiceTyping.pasteLabel")}
            />
          </SettingsRow>

          {pasteAutomatically ? (
            <SettingsRow
              label={t("settings.voice.voiceTyping.submitLabel")}
              description={t("settings.voice.voiceTyping.submitDescription")}
            >
              <Switch
                checked={submitAfterInsert}
                onCheckedChange={setSubmitAfterInsert}
                data-testid="voice-typing-submit"
                aria-label={t("settings.voice.voiceTyping.submitLabel")}
              />
            </SettingsRow>
          ) : null}

          <SettingsRow
            label={t("settings.voice.voiceTyping.trailingSpaceLabel")}
            description={t(
              "settings.voice.voiceTyping.trailingSpaceDescription",
            )}
          >
            <Switch
              checked={appendTrailingSpace}
              onCheckedChange={setAppendTrailingSpace}
              data-testid="voice-typing-trailing-space"
              aria-label={t("settings.voice.voiceTyping.trailingSpaceLabel")}
            />
          </SettingsRow>

          {pasteAutomatically ? (
            <SettingsRow
              label={t("settings.voice.voiceTyping.copyTranscriptLabel")}
              description={t(
                "settings.voice.voiceTyping.copyTranscriptDescription",
              )}
            >
              <Switch
                checked={copyLastTranscript}
                onCheckedChange={setCopyLastTranscript}
                data-testid="voice-typing-copy-transcript"
                aria-label={t("settings.voice.voiceTyping.copyTranscriptLabel")}
              />
            </SettingsRow>
          ) : null}

          <SettingsGroupDivider />

          <SettingsRow
            label={t("settings.voice.voiceTyping.fillerWordsLabel")}
            description={t("settings.voice.voiceTyping.fillerWordsDescription")}
          >
            <Switch
              checked={removeFillerWords}
              onCheckedChange={setRemoveFillerWords}
              data-testid="voice-typing-filler-words"
              aria-label={t("settings.voice.voiceTyping.fillerWordsLabel")}
            />
          </SettingsRow>

          <SettingsRow
            label={t("settings.voice.voiceTyping.spokenPunctuationLabel")}
            description={t(
              "settings.voice.voiceTyping.spokenPunctuationDescription",
            )}
          >
            <Switch
              checked={spokenPunctuation}
              onCheckedChange={setSpokenPunctuation}
              data-testid="voice-typing-spoken-punctuation"
              aria-label={t(
                "settings.voice.voiceTyping.spokenPunctuationLabel",
              )}
            />
          </SettingsRow>

          <SettingsRow
            label={t("settings.voice.voiceTyping.autoStopLabel")}
            description={t("settings.voice.voiceTyping.autoStopDescription")}
          >
            <Switch
              checked={autoStopSilence}
              onCheckedChange={setAutoStopSilence}
              data-testid="voice-typing-auto-stop"
              aria-label={t("settings.voice.voiceTyping.autoStopLabel")}
            />
          </SettingsRow>

          {autoStopSilence ? (
            <SettingsRow
              label={t("settings.voice.voiceTyping.autoStopDelayLabel")}
              description={t(
                "settings.voice.voiceTyping.autoStopDelayDescription",
              )}
            >
              <Select
                value={String(autoStopDelayMs)}
                onValueChange={(value) => setAutoStopDelayMs(Number(value))}
              >
                <SelectTrigger
                  size="sm"
                  className="w-44"
                  data-testid="voice-typing-auto-stop-delay"
                  aria-label={t(
                    "settings.voice.voiceTyping.autoStopDelayLabel",
                  )}
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1500">
                    {t("settings.voice.voiceTyping.seconds", {
                      seconds: "1.5",
                    })}
                  </SelectItem>
                  <SelectItem value="2500">
                    {t("settings.voice.voiceTyping.seconds", {
                      seconds: "2.5",
                    })}
                  </SelectItem>
                  <SelectItem value="4000">
                    {t("settings.voice.voiceTyping.seconds", { seconds: 4 })}
                  </SelectItem>
                  <SelectItem value="6000">
                    {t("settings.voice.voiceTyping.seconds", { seconds: 6 })}
                  </SelectItem>
                </SelectContent>
              </Select>
            </SettingsRow>
          ) : null}

          <SettingsGroupDivider />

          <SettingsRow
            label={t("settings.voice.voiceTyping.soundsLabel")}
            description={t("settings.voice.voiceTyping.soundsDescription")}
          >
            <Switch
              checked={soundEnabled}
              onCheckedChange={setSoundEnabled}
              data-testid="voice-typing-sounds"
              aria-label={t("settings.voice.voiceTyping.soundsLabel")}
            />
          </SettingsRow>

          {soundEnabled ? (
            <>
              <SettingsRow
                label={t("settings.voice.voiceTyping.soundsThemeLabel")}
                description={t(
                  "settings.voice.voiceTyping.soundsThemeDescription",
                )}
              >
                <div className="flex items-center gap-2">
                  <Select
                    value={soundTheme}
                    onValueChange={(value) =>
                      setSoundTheme(value as VoiceTypingSoundTheme)
                    }
                  >
                    <SelectTrigger
                      size="sm"
                      className="w-36"
                      data-testid="voice-typing-sound-theme"
                      aria-label={t(
                        "settings.voice.voiceTyping.soundsThemeLabel",
                      )}
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {VOICE_TYPING_SOUND_THEMES.map((theme) => (
                        <SelectItem key={theme} value={theme}>
                          {t(SOUND_THEME_LABEL[theme])}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    size="sm"
                    data-testid="voice-typing-sound-preview"
                    aria-label={t("settings.voice.voiceTyping.soundsPreview")}
                    onClick={() =>
                      previewVoiceTypingSound(soundTheme, soundVolume)
                    }
                  >
                    <Play className="size-3.5" />
                  </Button>
                </div>
              </SettingsRow>

              <SettingsRow
                label={t("settings.voice.voiceTyping.soundsVolumeLabel")}
                description={t(
                  "settings.voice.voiceTyping.soundsVolumeDescription",
                )}
              >
                <div className="flex items-center gap-3">
                  <Slider
                    value={[soundVolume]}
                    min={0}
                    max={1}
                    step={0.05}
                    onValueChange={([value]) => {
                      if (value !== undefined) setSoundVolume(value);
                    }}
                    className="w-36"
                    aria-label={t(
                      "settings.voice.voiceTyping.soundsVolumeLabel",
                    )}
                  />
                  <span className="w-9 text-right text-xs tabular-nums text-muted-foreground">
                    {Math.round(soundVolume * 100)}%
                  </span>
                </div>
              </SettingsRow>
            </>
          ) : null}

          <SettingsGroupDivider />

          <SettingsRow
            label={t("settings.voice.voiceTyping.preloadLabel")}
            description={t("settings.voice.voiceTyping.preloadDescription")}
          >
            <Switch
              checked={preloadModel}
              onCheckedChange={setPreloadModel}
              data-testid="voice-typing-preload"
              aria-label={t("settings.voice.voiceTyping.preloadLabel")}
            />
          </SettingsRow>

          <p className="pt-1 text-xs text-muted-foreground">
            {t("settings.voice.voiceTyping.engineNote")}
          </p>
        </>
      ) : null}
    </SettingsSection>
  );
}
